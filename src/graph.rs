//! Graph IR for fusekit.
//!
//! Arena-based representation: every Tensor and Op lives in a flat `Vec`
//! inside `Graph`, and edges are stored as `u32` indices rather than
//! references. This avoids self-referential borrow problems, keeps the
//! representation `Clone`/`Serialize`-friendly, and lets the compiler walk
//! the graph in topological order by simply iterating the vectors.

use std::fmt;

/// Index into `Graph::tensors`.
pub type TensorId = u32;

/// Index into `Graph::ops`.
pub type OpId = u32;

/// Element type of a tensor. Mirrors `mlx::core::Dtype::Val`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Bf16,
    Complex64,
    /// Anything we have not mapped yet, kept as a free-form string for
    /// diagnostics. Compilation must reject these explicitly.
    Other(&'static str),
}

impl Dtype {
    /// Convert the short MLX dtype name to our enum. Unknown names fall
    /// through to `Other("...")` rather than panicking; downstream passes
    /// can decide how strict to be.
    pub fn from_mlx_name(name: &str) -> Self {
        match name {
            "bool" => Dtype::Bool,
            "uint8" => Dtype::U8,
            "uint16" => Dtype::U16,
            "uint32" => Dtype::U32,
            "uint64" => Dtype::U64,
            "int8" => Dtype::I8,
            "int16" => Dtype::I16,
            "int32" => Dtype::I32,
            "int64" => Dtype::I64,
            "float16" => Dtype::F16,
            "float32" => Dtype::F32,
            "float64" => Dtype::F64,
            "bfloat16" => Dtype::Bf16,
            "complex64" => Dtype::Complex64,
            _ => Dtype::Other("unknown"),
        }
    }

    pub fn short(&self) -> &'static str {
        match self {
            Dtype::Bool => "b",
            Dtype::U8 => "u8",
            Dtype::U16 => "u16",
            Dtype::U32 => "u32",
            Dtype::U64 => "u64",
            Dtype::I8 => "i8",
            Dtype::I16 => "i16",
            Dtype::I32 => "i32",
            Dtype::I64 => "i64",
            Dtype::F16 => "f16",
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::Bf16 => "bf16",
            Dtype::Complex64 => "c64",
            Dtype::Other(_) => "?",
        }
    }
}

/// Op-specific attributes. Filled in lazily — Phase 1 leaves it at
/// `Unknown`, later phases populate concrete variants for the ops the
/// compiler actually understands.
#[derive(Debug, Clone)]
// The loader fills in concrete variants based on `RawRecord.arguments`.
// Ops whose only meaningful information is shape (Reshape, ExpandDims,
// Flatten, Unflatten, Squeeze, ...) can stay `Unknown` because the
// output tensor's shape is already in the IR.
#[allow(dead_code)]
pub enum Attrs {
    /// We have not extracted any structured parameters yet. The op's
    /// `name` field is still authoritative.
    Unknown,

    // ----- Reshape family -----
    Reshape {
        target_shape: Vec<i32>,
    },
    Transpose {
        axes: Vec<i32>,
    },
    Slice {
        start: Vec<i32>,
        stop: Vec<i32>,
        stride: Vec<i32>,
    },
    Concatenate {
        axis: i32,
    },
    Split {
        axis: i32,
    },
    Broadcast {
        target_shape: Vec<i32>,
    },

    // ----- Reductions -----
    Reduce {
        kind: ReduceKind,
        axes: Vec<i32>,
    },

    // ----- Normalization / activation -----
    RmsNorm {
        eps: f32,
    },
    LayerNorm {
        eps: f32,
    },
    Softmax {
        axis: i32,
    },

    // ----- Attention helpers (mlx::core::fast) -----
    Rope {
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        /// Optional compile-time `freqs` vector for traces where
        /// `mx.fast.rope` receives a small constant frequency tensor
        /// instead of deriving frequencies from `base`. This is still
        /// graph-semantic: the compiler only fills it by evaluating a
        /// pure constant producer subgraph.
        inline_freqs: Option<Vec<f32>>,
    },
    ScaledDotProductAttention {
        scale: f32,
    },

    // ----- Linear algebra -----
    AddMm {
        alpha: f32,
        beta: f32,
    },
    /// `QuantizedMatmul` parameters: bits (4 or 8) and group_size for
    /// the affine quant layout. Pattern matchers cross-reference these
    /// to pick the right dequant variant.
    Quantized {
        bits: u32,
        group_size: u32,
    },

    // ----- Iota -----
    Arange {
        start: f64,
        stop: f64,
        step: f64,
    },

    // ----- Indexing -----
    Gather {
        axes: Vec<i32>,
        slice_sizes: Vec<i32>,
    },

    // ----- Quantization (mlx::core::fast) -----
    Quantize {
        group_size: i32,
        bits: i32,
    },
    QuantizedMatmul {
        group_size: i32,
        bits: i32,
        transpose: bool,
    },

    // ----- Custom user-defined Metal kernel (mlx::fast::CustomKernel) -----
    /// mlx's `mx.fast.metal_kernel` primitive — a user-authored Metal
    /// kernel embedded directly in the model. `state()` from the C++
    /// primitive exposes name, source, grid, threadgroup, etc. We
    /// keep all of it so codegen can drop the source verbatim into
    /// our metallib and the runtime can dispatch via `name` with the
    /// captured grid/threadgroup.
    CustomKernel {
        name: String,
        source: String,
        grid: [u32; 3],
        threadgroup: [u32; 3],
        shared_memory: u32,
        ensure_row_contiguous: bool,
    },
}

// Same placeholder story as `Attrs` — wired in once the loader starts
// classifying reduce ops.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
    Any,
    All,
}

/// Captured value of a 0-dim trace constant. The runtime
/// auto-dispatcher reads these off external bindings to materialize
/// mlx-internal scalar uniforms (axis indices, broadcast dims, scale
/// factors, …) into per-kernel buffers without re-running the trace.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    Bool(bool),
    Int(i64),
    Float(f64),
}

/// Which forward-pass region an op belongs to. Layer-granular dispatch
/// builds one Metal ICB per `LayerTag`, so this is the partition key.
///
/// Ordering is `Embed < Layer(0) < Layer(1) < … < Post`, which makes
/// the propagation pass natural: an op that reads multiple producers
/// inherits the highest tag (= the layer that's actively producing,
/// not the one whose residual it's adding).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerTag {
    Embed,
    Layer(i32),
    Post,
}

/// A value flowing through the graph. Carries shape/dtype and, for
/// non-leaf tensors, the op that produced it.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Vec<i32>,
    pub dtype: Dtype,

    /// Op that produced this tensor. `None` means a leaf — typically a
    /// model input, weight, or constant.
    pub producer: Option<OpId>,

    /// Position within the producer's `outputs` list. Always 0 for
    /// single-output ops; non-zero only for multi-output ops like Split.
    /// Read once we land Split/MoE-router lowering — until then no
    /// consumer needs it, but keeping it in the IR avoids a schema bump
    /// later.
    #[allow(dead_code)]
    pub output_index: u32,

    /// Stable identity from the source MLX array, used during traversal
    /// to deduplicate shared subgraphs. Not meaningful after construction.
    #[allow(dead_code)]
    pub source_id: u64,

    /// Python attribute path for tensors that came from a model
    /// parameter (e.g. `model.layers.0.self_attn.q_proj.weight`).
    /// `None` for intermediates and for inputs we couldn't match to
    /// any tree-flattened parameter. Populated by `loader::lower` from
    /// the trace's `name_to_path` join.
    pub param_path: Option<String>,

    /// Captured value for 0-dim trace constants. `None` for any
    /// tensor that's not a scalar leaf.
    pub scalar_value: Option<ScalarValue>,
}

/// A node in the computation graph — one primitive operation.
#[derive(Debug, Clone)]
pub struct Op {
    pub id: OpId,
    /// Primitive name as reported by MLX (e.g. "Matmul", "fast::RMSNorm").
    pub name: String,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub attrs: Attrs,
    /// Region this op belongs to (which transformer layer, or the
    /// embed/post-norm pre/post-amble). Populated by
    /// `infer_layer_tags`. `None` when neither this op nor its
    /// producers ever touched a model parameter (e.g. ops that only
    /// consume the input token, or KV-cache-only ops on the first
    /// call).
    pub layer_id: Option<LayerTag>,
}

/// The graph itself: a flat arena of tensors and ops plus the root output.
#[derive(Debug, Clone)]
pub struct Graph {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    /// Final tensor of the computation (typically the model output).
    pub root: TensorId,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            tensors: Vec::new(),
            ops: Vec::new(),
            root: 0,
        }
    }

    /// Add a leaf tensor (input / weight / constant). Returns its id.
    pub fn add_input(&mut self, shape: Vec<i32>, dtype: Dtype, source_id: u64) -> TensorId {
        let id = self.tensors.len() as TensorId;
        self.tensors.push(Tensor {
            id,
            shape,
            dtype,
            producer: None,
            output_index: 0,
            source_id,
            param_path: None,
            scalar_value: None,
        });
        id
    }

    /// Add an op with its output tensors. Returns the new op's id and the
    /// ids of the freshly allocated output tensors. Inputs must already
    /// exist in the graph.
    pub fn add_op(
        &mut self,
        name: impl Into<String>,
        inputs: Vec<TensorId>,
        outputs: Vec<(Vec<i32>, Dtype, u64)>,
        attrs: Attrs,
    ) -> (OpId, Vec<TensorId>) {
        let op_id = self.ops.len() as OpId;

        let mut output_ids = Vec::with_capacity(outputs.len());
        for (idx, (shape, dtype, source_id)) in outputs.into_iter().enumerate() {
            let tid = self.tensors.len() as TensorId;
            self.tensors.push(Tensor {
                id: tid,
                shape,
                dtype,
                producer: Some(op_id),
                output_index: idx as u32,
                source_id,
                param_path: None,
                scalar_value: None,
            });
            output_ids.push(tid);
        }

        self.ops.push(Op {
            id: op_id,
            name: name.into(),
            inputs,
            outputs: output_ids.clone(),
            attrs,
            layer_id: None,
        });

        (op_id, output_ids)
    }

    pub fn tensor(&self, id: TensorId) -> &Tensor {
        &self.tensors[id as usize]
    }

    pub fn op(&self, id: OpId) -> &Op {
        &self.ops[id as usize]
    }

    pub fn set_root(&mut self, id: TensorId) {
        debug_assert!((id as usize) < self.tensors.len(), "root id out of range");
        self.root = id;
    }

    /// Graphviz DOT representation. Transformer graphs are highly
    /// repetitive (N identical layer blocks), so we detect the dominant
    /// repeating op-name period and render only one representative
    /// layer, annotated with `× count`. Cross-layer edges fold back onto
    /// the rendered layer, making the recurrence visible as a self-edge.
    pub fn to_dot(&self) -> String {
        use std::collections::HashSet;
        use std::fmt::Write;

        // Detect the repeating layer block.
        let block = find_repeating_block(&self.ops);

        // canonical[i] = the op id we display in place of op i.
        // For ops outside the block, that's i itself. For ops inside,
        // it's the equivalent op in the first iteration.
        let canonical: Vec<u32> = (0..self.ops.len() as u32)
            .map(|i| match block {
                Some((start, period, count))
                    if (i as usize) >= start + period && (i as usize) < start + period * count =>
                {
                    (start + ((i as usize - start) % period)) as u32
                }
                _ => i,
            })
            .collect();

        // The set of op ids that survive the collapse (i.e. ops whose
        // canonical id equals themselves).
        let kept: HashSet<u32> = canonical
            .iter()
            .enumerate()
            .filter(|(i, c)| **c == *i as u32)
            .map(|(_, c)| *c)
            .collect();

        let mut s = String::new();
        s.push_str("digraph fusekit {\n");
        s.push_str("  rankdir=TB;\n");
        s.push_str("  graph [splines=ortho, nodesep=0.25, ranksep=0.4];\n");
        s.push_str("  node  [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, fillcolor=\"#eef3fb\", color=\"#5b7fb0\"];\n");
        s.push_str("  edge  [fontname=\"Helvetica\", fontsize=8, color=\"#7a8aa0\"];\n");

        // Op nodes, with the repeated layer wrapped in a labelled cluster.
        let (layer_start, layer_end, layer_count) = match block {
            Some((s_, p, c)) => (s_, s_ + p, c),
            None => (usize::MAX, usize::MAX, 0),
        };

        // Emit ops before the cluster.
        for op in self.ops.iter().take(layer_start.min(self.ops.len())) {
            if !kept.contains(&op.id) {
                continue;
            }
            let _ = writeln!(
                s,
                "  op{} [label=\"#{} {}\"];",
                op.id,
                op.id,
                dot_quote(&op.name)
            );
        }

        // The cluster of one layer.
        if block.is_some() {
            let _ = writeln!(s, "  subgraph cluster_layer {{");
            let _ = writeln!(s, "    label=\"layer block × {}\";", layer_count);
            s.push_str("    style=\"rounded,dashed\";\n");
            s.push_str("    color=\"#888\";\n");
            s.push_str("    fontname=\"Helvetica\";\n");
            s.push_str("    fontsize=12;\n");
            for op in &self.ops[layer_start..layer_end] {
                let _ = writeln!(
                    s,
                    "    op{} [label=\"#{} {}\"];",
                    op.id,
                    op.id,
                    dot_quote(&op.name)
                );
            }
            s.push_str("  }\n");
        }

        // Emit ops after the cluster (postamble).
        let post_start = if block.is_some() {
            layer_start + layer_count * (layer_end - layer_start)
        } else {
            0
        };
        if block.is_some() {
            for op in &self.ops[post_start..] {
                if !kept.contains(&op.id) {
                    continue;
                }
                let _ = writeln!(
                    s,
                    "  op{} [label=\"#{} {}\"];",
                    op.id,
                    op.id,
                    dot_quote(&op.name)
                );
            }
        } else {
            // No block detected — render everything we didn't already emit.
            for op in &self.ops {
                if !kept.contains(&op.id) {
                    continue;
                }
                let _ = writeln!(
                    s,
                    "  op{} [label=\"#{} {}\"];",
                    op.id,
                    op.id,
                    dot_quote(&op.name)
                );
            }
        }

        // Single virtual "inputs" node for leaf tensors.
        let has_leaf = self
            .ops
            .iter()
            .any(|op| op.inputs.iter().any(|&t| self.tensor(t).producer.is_none()));
        if has_leaf {
            s.push_str("  inputs [label=\"inputs / weights\", shape=ellipse, fillcolor=\"#fff4d6\", color=\"#c89028\"];\n");
        }

        // Edges. Collapse via the canonical map and dedupe — same src/dst
        // pair across multiple iterations of the block becomes one edge.
        let mut seen: HashSet<(i64, u32, String)> = HashSet::new();
        for op in &self.ops {
            let dst = canonical[op.id as usize];
            for &tid in &op.inputs {
                let t = self.tensor(tid);
                let edge_label = dot_shape_label(&t.shape, t.dtype);
                let src: i64 = match t.producer {
                    Some(p) => canonical[p as usize] as i64,
                    None => -1,
                };
                let key = (src, dst, edge_label.clone());
                if !seen.insert(key) {
                    continue;
                }
                if src < 0 {
                    let _ = writeln!(
                        s,
                        "  inputs -> op{} [label=\"{}\", style=dashed];",
                        dst, edge_label
                    );
                } else if src as u32 == dst {
                    // Self-loop introduced by the collapse — that is the
                    // residual / layer-to-layer feedback. Draw it once,
                    // arched, with a distinctive style.
                    let _ = writeln!(
                        s,
                        "  op{} -> op{} [label=\"{}  (×{})\", color=\"#b85c3a\", style=dashed, constraint=false];",
                        src, dst, edge_label, layer_count.saturating_sub(1).max(1)
                    );
                } else {
                    let _ = writeln!(s, "  op{} -> op{} [label=\"{}\"];", src, dst, edge_label);
                }
            }
        }

        // Highlight the root.
        if let Some(root_op) = self.tensor(self.root).producer {
            let canon_root = canonical[root_op as usize];
            let _ = writeln!(
                s,
                "  op{} [fillcolor=\"#d4f0d4\", color=\"#3e8a3e\", penwidth=2];",
                canon_root
            );
        }

        s.push_str("}\n");
        s
    }

    /// Human-readable dump, useful for tests and debug commands.
    pub fn pretty(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "graph: {} tensors, {} ops, root=t{}\n",
            self.tensors.len(),
            self.ops.len(),
            self.root
        ));

        for t in &self.tensors {
            let origin = match t.producer {
                Some(op_id) => {
                    let op = &self.ops[op_id as usize];
                    let args: Vec<String> = op.inputs.iter().map(|i| format!("t{i}")).collect();
                    format!("{}({})", op.name, args.join(", "))
                }
                None => "input".to_string(),
            };
            s.push_str(&format!(
                "  t{:<4} {:<10} {:?}  <- {}\n",
                t.id,
                t.dtype.short(),
                t.shape,
                origin
            ));
        }
        s
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.pretty())
    }
}

/// Detect the longest contiguous run of identical op-name blocks of
/// some period P in the op sequence. Returns `(start, period, count)`
/// such that `ops[start..start+period]` equals each of the `count`
/// blocks starting at `start, start+period, ..., start+(count-1)*period`.
///
/// Used to collapse repeated transformer layers in the DOT export.
fn find_repeating_block(ops: &[Op]) -> Option<(usize, usize, usize)> {
    let n = ops.len();
    if n < 30 {
        return None;
    }
    let names: Vec<&str> = ops.iter().map(|o| o.name.as_str()).collect();
    let max_period = (n / 3).min(500);

    let mut best: Option<(usize, usize, usize)> = None;

    for period in 10..=max_period {
        // Longest contiguous run of indices i where names[i] == names[i+period].
        let mut run_start = 0usize;
        let mut run_len = 0usize;
        let mut best_start = 0usize;
        let mut best_len = 0usize;
        for i in 0..n - period {
            if names[i] == names[i + period] {
                if run_len == 0 {
                    run_start = i;
                }
                run_len += 1;
                if run_len > best_len {
                    best_len = run_len;
                    best_start = run_start;
                }
            } else {
                run_len = 0;
            }
        }
        if best_len < 2 * period {
            continue;
        }
        let count = best_len / period + 1;
        // Verify the block is byte-identical across all `count` copies.
        let block = &names[best_start..best_start + period];
        let mut verified = true;
        for k in 1..count {
            let off = best_start + k * period;
            if &names[off..off + period] != block {
                verified = false;
                break;
            }
        }
        if !verified {
            continue;
        }
        let score = count * period;
        let take = match best {
            None => true,
            Some((_, p, c)) => score > p * c,
        };
        if take {
            best = Some((best_start, period, count));
        }
    }
    best
}

fn dot_quote(s: &str) -> String {
    s.replace('"', "\\\"").replace('\\', "\\\\")
}

fn dot_shape_label(shape: &[i32], dtype: Dtype) -> String {
    if shape.is_empty() {
        return dtype.short().to_string();
    }
    let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("{} {}", parts.join("x"), dtype.short())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_build_and_walk() {
        // y = matmul(x, W) + b
        let mut g = Graph::new();
        let x = g.add_input(vec![1, 1024], Dtype::F32, 1);
        let w = g.add_input(vec![1024, 4096], Dtype::F32, 2);
        let b = g.add_input(vec![4096], Dtype::F32, 3);

        let (_, mm_out) = g.add_op(
            "Matmul",
            vec![x, w],
            vec![(vec![1, 4096], Dtype::F32, 100)],
            Attrs::Unknown,
        );

        let (_, add_out) = g.add_op(
            "Add",
            vec![mm_out[0], b],
            vec![(vec![1, 4096], Dtype::F32, 101)],
            Attrs::Unknown,
        );

        g.set_root(add_out[0]);

        assert_eq!(g.tensors.len(), 5);
        assert_eq!(g.ops.len(), 2);
        assert_eq!(g.tensor(g.root).producer, Some(1));
        let producer = g.op(g.tensor(g.root).producer.unwrap());
        assert_eq!(producer.name, "Add");
        assert_eq!(producer.inputs, vec![mm_out[0], b]);
    }
}
