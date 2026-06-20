// Scaffold module — items go unused until pattern matchers and codegen wire in.
#![allow(dead_code)]

//! Stage 2 IR — Fusion graph.
//!
//! ```text
//!   Stage 1 (Graph)              Stage 2 (FusionGraph)
//!   ─────────────────            ─────────────────────
//!   Op   Op   Op   Op            ┌──────────────┐
//!   │    │    │    │             │  Fusion #0   │  ← N ops collapsed into
//!   tensors flow                 │              │    one kernel function
//!         │           pattern    │ inputs:  ... │
//!         ▼           match      │ outputs: ... │
//!                  ──────────►   │ parallel:... │
//!                                └──────────────┘
//!                                        ▼
//!                                ┌──────────────┐
//!                                │  Fusion #1   │
//!                                └──────────────┘
//! ```
//!
//! A `Fusion` is the unit that becomes ONE MSL kernel function. It
//! carries the boundary tensors (shape + role label, no pointers) plus
//! the structured parameters needed for codegen. Pointers and grid
//! decisions belong to the Stage 3 `Task` IR.

use crate::graph::{Dtype, OpId, ScalarValue, TensorId};

pub type FusionId = u32;

/// The full Stage 2 IR. `fusions` are in topological order so consumers
/// of a fusion's outputs always have a higher id.
#[derive(Debug)]
pub struct FusionGraph {
    pub fusions: Vec<Fusion>,
    /// Model-wide architecture constants captured at fuse time so
    /// downstream stages don't need to reparse `config.json`.
    pub meta: ModelMeta,
}

/// One unit of fused computation — one MSL kernel function.
#[derive(Debug, Clone)]
pub struct Fusion {
    pub id: FusionId,
    pub kind: FusionKind,
    /// Original ops captured into this fusion. Useful for debugging /
    /// regenerating MSL on demand; not load-bearing for codegen.
    pub op_ids: Vec<OpId>,
    /// Tensors flowing IN from outside the fusion. Order matters —
    /// matches the order codegen lays out kernel arguments.
    pub inputs: Vec<FusionBoundary>,
    /// Tensors flowing OUT of the fusion. Same ordering contract.
    pub outputs: Vec<FusionBoundary>,
    /// Structured per-pattern parameters. Codegen pattern-matches on
    /// this enum to emit the right kernel body.
    pub attrs: FusionAttrs,
    /// Axes along which this fusion can be safely partitioned across
    /// multiple tasks at Stage 3. Empty = run as a single task.
    pub parallel_axes: Vec<ParallelAxis>,
    /// Forward-pass region this fusion belongs to. Trace-bound
    /// fusions get tagged via op_ids propagation; synthetic
    /// (utility) fusions whose `op_ids` is empty set this directly
    /// so the runtime can group per-layer dispatch correctly.
    pub layer_hint: Option<crate::graph::LayerTag>,
}

/// Boundary tensor — shape/dtype + a role label codegen reads.
///
/// Roles are pattern-specific string literals (e.g. "x", "W_q",
/// "kv_cache"). The codegen template picks each input by role, not by
/// position, so adding a new boundary tensor to a pattern doesn't break
/// downstream code as long as the role names are stable.
#[derive(Debug, Clone)]
pub struct FusionBoundary {
    /// Original Stage 1 tensor this boundary references. Used to wire
    /// Fusion outputs to subsequent Fusion inputs in Stage 3.
    pub source: TensorId,
    pub role: &'static str,
    pub shape: Vec<i32>,
    pub dtype: Dtype,
    /// Captured value for scalar constant boundaries. General fusion
    /// may remap constants into merged kernels, so codegen cannot
    /// depend on looking them up through the original graph alone.
    pub scalar_value: Option<ScalarValue>,
}

/// What kind of fusion this is. Codegen branches on this to pick the
/// right MSL template. `Single` is the universal fallback for ops we
/// haven't taught a pattern matcher about.
#[derive(Debug, Clone)]
pub enum FusionKind {
    /// One op, no fusion. Codegen dispatches to a per-op template (or
    /// in a future phase, calls back into MLX's own kernel).
    Single(String),

    // ----- Pattern-matched fused kernels -----
    /// RMSNorm → linear projection (Q/K/V or any single projection).
    RmsNormProj,
    /// RMSNorm → three linear projections (Q, K, V) sharing the norm.
    RmsNormQkv,
    /// SDPA decode path — Q · Kᵀ → scale → softmax → · V.
    /// Matched as `Matmul → Multiply → Softmax → Matmul` chain in MLX
    /// traces. The kernel is decode-only (Q has L=1) for now; prefill
    /// gets a separate kind once we land its FlashAttention emitter.
    SdpaDecode,
    /// SDPA prefill — FlashAttention-style tiled attention.
    SdpaPrefill,
    /// Attention output projection + residual add.
    AttnOutResidual,
    /// Generic `Linear + residual Add` — fuses any Matmul whose only
    /// downstream consumer is an Add receiving the matmul output and
    /// one external residual tensor. Subsumes both `o_proj + residual`
    /// and `down_proj + residual` cases.
    LinearResidual,
    /// SwiGLU activation: `sigmoid(gate) * gate * up`. No projections —
    /// just the activation glue between gate/up and down_proj.
    SwigluActivation,
    /// Pre-MLP RMSNorm + (gate, up) projections only. The downstream
    /// SwiGLU activation, down_proj, and residual stay as separate
    /// fusions for now.
    RmsNormGateUp,
    /// Pre-MLP RMSNorm + (gate, up) projections + SwiGLU activation in
    /// one kernel. The standalone `RmsNormGateUp` materializes gate
    /// and up to global memory; this variant computes them in
    /// per-thread registers and applies `sigmoid(gate) · gate · up`
    /// before writing only the SwiGLU output. Saves 2·N_INT·bf16 of
    /// global memory traffic per layer (~19KB on Qwen-0.5B).
    RmsNormGateUpSwiglu,
    /// Full post-attn MLP block in one kernel:
    /// `RMSNorm → (gate, up) → SwiGLU → down_proj + residual`. The
    /// SwiGLU output stays in threadgroup memory (4864 floats ≈ 19KB
    /// on Qwen-0.5B) so the down-projection's reduction reads from
    /// SMEM instead of round-tripping through DRAM. The residual
    /// addend is the same `h_attn` that was the kernel's `x` input,
    /// so we read it once into registers and reuse for both the norm
    /// and the final add.
    RmsNormGateUpSwigluDown,
    /// Post-attn RMSNorm + SwiGLU MLP (gate · sigmoid · up → down) +
    /// residual. End-to-end MLP block; not yet emitted.
    RmsNormSwigluMlp,
    /// Final RMSNorm + LM-head projection in one kernel: takes the
    /// post-final-layer hidden state and emits logits directly. Saves
    /// a hidden-sized DRAM round-trip between the norm and the
    /// (very large) vocab projection.
    FinalRmsNormLmHead,
    /// Decode-time RoPE applied to Q in place. Reads `offset` (current
    /// position) from a uniform and rotates each `head_dim/2` pair.
    RopeQDecode,
    /// Decode-time RoPE on K (in place) + store K and V into the
    /// (1, n_kv, capacity, head_dim) KV cache at `offset`.
    RopeKvCacheStoreDecode,
    /// Decode-time store for attention paths whose current K/V are
    /// already materialized in scratch in cache layout order. This
    /// covers MLA-style attention where K has a wider QK dimension
    /// than V and RoPE has already been applied by trace-preserved
    /// ops.
    KvCacheStoreDecode,
    /// Decode-time Qwen3Next-style query preparation. The trace's
    /// q projection carries `[query, gate]` per head; this applies
    /// per-head RMSNorm + partial RoPE to query and materializes the
    /// flat gate vector consumed after SDPA.
    RmsNormRopeQGateDecode,
    /// Decode-time Qwen3Next-style K/V cache store. K gets per-head
    /// RMSNorm + partial RoPE before store; V is copied to the value
    /// cache at the current offset.
    RmsNormRopeKvCacheStoreDecode,
    /// Decode-time single-token embedding lookup. Writes the
    /// embedding row to a pre-allocated, stable-address buffer so
    /// the rest of the layer chain (and the ICB that pre-encodes it)
    /// reads from a fixed pointer.
    EmbedLookupDecode,
    /// Decode-time state update for state-space/convolutional caches:
    /// shift an `ArraysCache` state window left by one token and
    /// append the current token projection. This is intentionally
    /// semantic (cache slot + window length), not model-name based.
    ArraysCacheShiftAppendDecode,
    /// Single-TG argmax over the LM-head logits, writing the winning
    /// vocab index into a small uniform so the next decode step
    /// can read it without a CPU round-trip.
    ArgmaxLogitsDecode,
    /// Multi-TG argmax scan over the LM-head logits. Each TG handles
    /// a slice of `vocab/N_TGS` elements and writes its (max, idx)
    /// pair to a tiny scratch buffer. `ArgmaxLogitsFinalize` reduces
    /// the scratch in one TG. Activated via FUSEKIT_ARGMAX_MULTI_TG=1.
    /// Splits the single-TG bandwidth-bound vocab scan into M parallel
    /// scans, then finalizes in O(N_TGS) entries.
    ArgmaxLogitsScan,
    /// Single-TG reduce of per-TG argmax scratch → next_tok.
    /// Companion of `ArgmaxLogitsScan`.
    ArgmaxLogitsFinalize,
    /// Op-semantic general fusion product: an ordered chain of
    /// pointwise / view / generator / reduction-tail ops merged into
    /// one MSL kernel by the Stage 3 general-fusion pass. Each step's
    /// intermediate value stays in registers (no DRAM round-trip
    /// between the steps in the chain). The chain itself is opaque to
    /// the pattern matchers — it's what's left over after pattern
    /// fusion plus the general-fusion pass folds neighbours together.
    FusedChain,
    /// Structural quantized projection fanout: multiple
    /// `QuantizedMatmul` siblings reading the same activation row are
    /// emitted as one dispatch. Admission is by shared input, quant
    /// metadata, shape, and budget, not by model-name pattern.
    QmvFanout,
    /// Structural gate/up quantized projection with a SwiGLU tail:
    /// two compatible `QuantizedMatmul` siblings reading the same
    /// activation row feed `sigmoid(gate) * gate * up`. The emitter
    /// computes both projections in registers and materializes only
    /// the activation output.
    QmvSwiglu,
    /// Structural routed-expert gate/up projection with a SwiGLU tail:
    /// two compatible `GatherQMM` siblings reading the same activation
    /// row and route/expert indices feed `sigmoid(gate) * gate * up`.
    /// This is admitted by op/dataflow/shape semantics, not by model
    /// name, and covers top-k MoE decode paths.
    RoutedQmvSwiglu,
    /// Structural routed-expert projection plus route weighted sum:
    /// a `GatherQMM` over selected expert outputs immediately feeds
    /// route-weight multiply/reduce. The emitter computes each output
    /// dim as `sum_r weight_r * expert_matvec_r` and materializes only
    /// the reduced hidden row, avoiding the `[top_k, hidden]` scratch
    /// round-trip. Admission is by routed QMM and reduce semantics.
    RoutedQmvReduce,
    /// Per-layer megakernel — composes the full transformer-block
    /// pipeline (norm → QKV → RoPE → SDPA → o_proj+residual → norm →
    /// gate/up → swiglu → down_proj+residual) into ONE MSL function
    /// via the piece composer. Removes ~10 DRAM round-trips per layer
    /// vs the standard 7-kernel layer dispatch. Routed through the
    /// layer-block matcher, not through pattern matching.
    LayerBlock,
}

/// Per-weight quantization metadata. Pattern matchers fill this in
/// when the source op is a `QuantizedMatmul`; codegen branches on
/// `Option<QuantInfo>` to pick the right inner loop.
///
/// mlx affine quantization layout (verified against `mx.quantized_matmul`):
/// ```text
///   packed weight  uint32, shape (out, in/8)         -- 8 int4 lanes per word, LSB-first
///   scales         fp16,   shape (out, in/group_size)
///   biases         fp16,   shape (out, in/group_size)
///   dequant: w[i,j] = uint4(packed[i,j/8], j%8) * scales[i, j/gs] + biases[i, j/gs]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QuantInfo {
    pub bits: u32,
    pub group_size: u32,
}

/// Decode-only consumers that can be sunk into the semantic QKV
/// projection node. Admission is by graph/runtime dataflow: the layer
/// has the standard synthetic Q RoPE and K/V cache-store consumers, so
/// the QKV emitter may write roped Q directly and store K/V into the
/// live cache without materializing K/V scratch.
#[derive(Debug, Clone, Copy)]
pub struct QkvDecodeSinks {
    pub theta: f32,
}

/// RoPE parameters captured from `mx.fast.rope` when a trace-local
/// RoPE can be sunk into a semantic consumer. This is op-level
/// metadata, not a model-name pattern.
#[derive(Debug, Clone)]
pub struct RopeInfo {
    pub dims: u32,
    pub traditional: bool,
    pub base: f32,
    pub scale: f32,
    pub inline_freqs: Option<Vec<f32>>,
}

/// Pattern-specific parameters, populated by the matcher. Codegen
/// `match`es on this to fetch the right values without revisiting the
/// Stage 1 IR.
#[derive(Debug, Clone)]
pub enum FusionAttrs {
    /// Stash the underlying op's typed attrs so the single emitter can
    /// see RoPE.theta, Softmax.axis, Reshape.target_shape, etc. without
    /// re-traversing the original Graph. `op_name` mirrors
    /// `FusionKind::Single(name)` for convenience at the emit site.
    Single {
        op_name: String,
        op_attrs: crate::graph::Attrs,
    },
    RmsNormProj {
        eps: f32,
        quant: Option<QuantInfo>,
    },
    RmsNormQkv {
        eps: f32,
        quant: Option<QuantInfo>,
        decode_sinks: Option<QkvDecodeSinks>,
    },
    SdpaDecode {
        scale: f32,
        trace_q: bool,
        /// Split-Q input layout for trace-bound attention. When set,
        /// `inputs[0]` and `inputs[1]` are concatenated along the last
        /// dimension to form logical Q; SDPA reads both directly instead
        /// of materializing a Concatenate scratch row.
        q_split: Option<u32>,
        /// The split Q inputs are last-dimension slices of the same
        /// pre-transpose packed source laid out as `[B,S,H,Q0+Q1]`.
        /// SDPA indexes that source directly as `(query_row, head, dim)`.
        q_transpose_split: bool,
        /// Optional RoPE to apply to Q1 before it participates in logical
        /// Q. Used when `RoPE(q1) -> Concatenate -> SdpaDecode.Q` is
        /// dataflow-isolated.
        q1_rope: Option<RopeInfo>,
        mask: AttentionMask,
    },
    SdpaPrefill {
        scale: f32,
        mask: AttentionMask,
    },
    AttnOutResidual,
    LinearResidual {
        quant: Option<QuantInfo>,
    },
    SwigluActivation,
    RmsNormGateUp {
        eps: f32,
        quant: Option<QuantInfo>,
    },
    RmsNormGateUpSwiglu {
        eps: f32,
        quant: Option<QuantInfo>,
    },
    RmsNormGateUpSwigluDown {
        eps: f32,
        quant: Option<QuantInfo>,
    },
    RmsNormSwigluMlp {
        eps: f32,
    },
    FinalRmsNormLmHead {
        eps: f32,
        quant: Option<QuantInfo>,
    },
    RopeQDecode {
        theta: f32,
    },
    RopeKvCacheStoreDecode {
        theta: f32,
    },
    KvCacheStoreDecode {
        /// Optional RoPE applied to split-K part1 while writing the
        /// current token into the cache. This represents graph-local
        /// `RoPE(k_part1)` sunk into the semantic cache-store primitive;
        /// admission is by dataflow/attrs, not model name.
        k1_rope: Option<RopeInfo>,
        /// Optional direct read from the pre-transpose packed K/V source.
        /// When set, split-K input0 and V input2 alias the same
        /// `[B, S, H, K0+V]` tensor and the value stores start at this
        /// K0 dimension. Admission is by TransposeSplit axes, tensor
        /// shapes, and cache-store semantic roles, not by model name.
        kv_transpose_split: Option<u32>,
    },
    RmsNormRopeQGateDecode {
        eps: f32,
        theta: f32,
        rope_dims: u32,
    },
    RmsNormRopeKvCacheStoreDecode {
        eps: f32,
        theta: f32,
        rope_dims: u32,
    },
    EmbedLookupDecode {
        /// Optional scalar applied immediately after token embedding.
        /// Some model families scale embeddings by sqrt(hidden_size);
        /// this is inferred from the captured graph as
        /// `Gather(embed_tokens, token) -> Multiply(_, scalar)`, not
        /// from model names.
        output_scale: f32,
    },
    ArraysCacheShiftAppendDecode {
        cache_index: u32,
        state_len: u32,
        width: u32,
    },
    ArgmaxLogitsDecode,
    /// Multi-TG argmax scan: each TG handles vocab/N_TGS slice.
    ArgmaxLogitsScan {
        n_tgs: u32,
    },
    /// Single-TG finalize over `ArgmaxLogitsScan`'s scratch.
    ArgmaxLogitsFinalize {
        n_tgs: u32,
    },
    /// Stage 3 product: ordered op chain whose intermediates live in
    /// registers. Each `ChainStep` references prior steps' outputs by
    /// step-index or external buffer-input by buffer-index.
    FusedChain {
        steps: Vec<ChainStep>,
        /// External buffer slots whose logical shape is the trace's
        /// transposed `[K, N]` view, but whose runtime binding is the
        /// original model parameter laid out as `[N, K]`. The emitter
        /// reads these slots with swapped indexing to avoid materializing
        /// dense weight-transpose scratch buffers at decode time.
        transposed_weight_slots: Vec<u32>,
    },
    QmvFanout {
        targets: Vec<QmvFanoutTarget>,
    },
    QmvSwiglu {
        gate: QmvFanoutTarget,
        up: QmvFanoutTarget,
    },
    RoutedQmvSwiglu {
        gate: QmvFanoutTarget,
        up: QmvFanoutTarget,
        pre_scale_route_weight: bool,
    },
    RoutedQmvReduce {
        target: QmvFanoutTarget,
        input_preweighted: bool,
        residual_linear: Option<QmvFanoutTarget>,
        has_extra_residual: bool,
    },
    /// Per-layer megakernel attrs. Carries the layer index (used for
    /// per-layer cache routing at the runtime) plus the shared per-block
    /// scalars the contained pieces read: rmsnorm `eps`, RoPE `theta`,
    /// weight quant info (uniform across the layer's int4 weights).
    LayerBlock {
        layer_idx: i32,
        eps: f32,
        theta: f32,
        quant: Option<QuantInfo>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct QmvFanoutTarget {
    pub k_dim: u32,
    pub out_dim: u32,
    pub active_rows: u64,
    pub bits: u32,
    pub group_size: u32,
}

/// One node of a [`FusionAttrs::FusedChain`]. The general-fusion pass
/// produces these in topological order; the emitter walks them top to
/// bottom, declaring `T rN = ...` per step.
///
/// Each step's `inputs` list resolves each operand to either an
/// *external* fusion-input buffer (read from DRAM via the kernel's
/// buffer slot N) or an *internal* register holding an earlier step's
/// output.
#[derive(Debug, Clone)]
pub struct ChainStep {
    /// Op name as it appeared in the original mlx trace ("Add",
    /// "Sigmoid", ...).
    pub op_name: String,
    /// Op-typed attrs (carries e.g. RmsNorm.eps, Softmax.axis).
    pub op_attrs: crate::graph::Attrs,
    /// One entry per op input, in mlx's argument order.
    pub inputs: Vec<ChainOperand>,
    /// Output tensor shape, for the emitter to size loops/indexing.
    pub output_shape: Vec<i32>,
    /// Output dtype of this original op. General fusion keeps scalar
    /// intermediates in registers, but MLX still rounds at dtype
    /// boundaries such as bf16 Sigmoid -> bf16 Multiply.
    pub output_dtype: Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainOperand {
    /// Read from kernel `buffer(N)` — an external input to the fusion.
    External { buffer_index: u32 },
    /// Read from the in-register output of step `step_index`.
    Register { step_index: u32 },
}

/// SDPA masking variants. Drives both kernel selection and which
/// indices are computed at runtime.
///
/// ```text
///   None      causal=false. Used during cross-attention or non-LM.
///   Causal    standard lower-triangular. Decode + prefill default.
///   Prefix    causal beyond a fixed prefix (system prompt) length.
///   Tree      tree-attention for parallel speculative branches.
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AttentionMask {
    None,
    Causal,
    Prefix(u32),
    Tree,
}

/// A dimension a fusion can be sliced along without inter-task
/// communication. The Stage 3 scheduler picks one of these to map onto
/// a grid axis.
///
/// `size` is the dimension's extent (e.g. num_heads = 24). `on` points
/// at which boundary tensor + which axis carries this dimension, so
/// codegen knows how to slice input pointers per-threadgroup.
#[derive(Debug, Clone)]
pub struct ParallelAxis {
    pub name: &'static str,
    pub size: u32,
    pub on: AxisRef,
}

/// "axis `dim` of `boundary`" — disambiguates when multiple boundary
/// tensors share the same axis (e.g. batch on both inputs and outputs).
#[derive(Debug, Clone)]
pub struct AxisRef {
    pub boundary_kind: BoundaryKind,
    pub boundary_index: u32,
    pub dim: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryKind {
    Input,
    Output,
}

/// Architecture-level constants pulled from `config.json` at capture
/// time. Everything here is compile-time constant for one model — bake
/// these into MSL string literals rather than passing as uniforms.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub model_type: String,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_kv_shared_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    /// `quantization.group_size` from config, or None for unquantized.
    pub quant_group_size: Option<u32>,
    /// `quantization.bits`, or None for unquantized.
    pub quant_bits: Option<u32>,
    /// Qwen-style native MTP head layer count.
    pub mtp_num_hidden_layers: u32,
    /// DeepSeek/GLM/MiMo/Nemotron-style next-token predictor count.
    pub num_nextn_predict_layers: u32,
    /// Resolved native-MTP metadata from config plus weight files.
    pub mtp: MtpInfo,
    /// Activation/compute dtype as the model runs. Drives whether
    /// emitted MSL uses `bfloat` or `half` for tensors that flow
    /// between kernels (norm weights, biases, activations, KV cache).
    /// Weight-only quantized models inherit their compute dtype from
    /// the un-quantized side: Qwen2.5-bf16 → Bf16, Qwen2.5-4bit (whose
    /// dequant lands as fp16) → F16.
    pub activation_dtype: Dtype,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MtpInfo {
    pub format: MtpFormat,
    pub declared_layers: u32,
    pub max_depth: Option<u32>,
    pub has_weights: bool,
    pub source: String,
}

impl MtpInfo {
    pub fn none() -> Self {
        Self {
            format: MtpFormat::None,
            declared_layers: 0,
            max_depth: None,
            has_weights: false,
            source: "none".to_string(),
        }
    }

    pub fn is_declared(&self) -> bool {
        self.format != MtpFormat::None
    }
}

impl Default for MtpInfo {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtpFormat {
    None,
    Mtplx,
    QwenNative,
    NextN,
}

impl ModelMeta {
    /// Load a `config.json` and extract the fields we need for codegen.
    ///
    /// Hugging Face model configs come in two layouts:
    ///   - flat: all fields at top level (older models like Qwen2.5)
    ///   - nested: text-portion fields under `text_config` or a
    ///     `model_type`-named sub-object (newer multimodal models
    ///     like Qwen3.6 with `qwen3_5_text` sub-config)
    /// We probe in that order and merge so callers don't care which
    /// shape the model uses.
    pub fn from_config_file(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let mut root: serde_json::Value =
            serde_json::from_str(&text).context("parse config.json")?;
        augment_missing_text_meta_from_safetensors(path, &mut root);
        let mut meta = Self::from_value(&root)?;
        if meta.quant_bits.is_some() {
            if let Some(dtype) = infer_quant_activation_dtype_from_safetensors(path) {
                meta.activation_dtype = dtype;
            }
        }
        meta.mtp = detect_mtp_info(
            path,
            &root,
            meta.mtp_num_hidden_layers,
            meta.num_nextn_predict_layers,
        );
        Ok(meta)
    }

    pub fn from_value(root: &serde_json::Value) -> anyhow::Result<Self> {
        // Find the inner text-model object. Either a `text_config`
        // sibling, or any sub-object whose own `model_type` ends with
        // "_text". Falls back to root when the config is already flat.
        let text_cfg = root
            .get("text_config")
            .or_else(|| {
                root.as_object()?.values().find(|v| {
                    v.get("model_type")
                        .and_then(|m| m.as_str())
                        .map(|s| s.ends_with("_text"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(root);

        let pick = |k: &str| -> Option<u64> { text_cfg.get(k).and_then(|v| v.as_u64()) };
        let pick_root = |k: &str| -> Option<u64> { root.get(k).and_then(|v| v.as_u64()) };

        let hidden_size = pick("hidden_size")
            .or_else(|| pick_root("hidden_size"))
            .ok_or_else(|| anyhow::anyhow!("config missing hidden_size"))?
            as u32;
        let num_attention_heads = pick("num_attention_heads")
            .or_else(|| pick_root("num_attention_heads"))
            .ok_or_else(|| anyhow::anyhow!("config missing num_attention_heads"))?
            as u32;
        let num_key_value_heads = pick("num_key_value_heads")
            .or_else(|| pick_root("num_key_value_heads"))
            .unwrap_or(num_attention_heads as u64) as u32;
        let num_hidden_layers = pick("num_hidden_layers")
            .or_else(|| pick_root("num_hidden_layers"))
            .ok_or_else(|| anyhow::anyhow!("config missing num_hidden_layers"))?
            as u32;
        let num_kv_shared_layers = pick("num_kv_shared_layers")
            .or_else(|| pick_root("num_kv_shared_layers"))
            .unwrap_or(0) as u32;
        let intermediate_size = pick("intermediate_size")
            .or_else(|| pick_root("intermediate_size"))
            .ok_or_else(|| anyhow::anyhow!("config missing intermediate_size"))?
            as u32;
        let vocab_size = pick("vocab_size")
            .or_else(|| pick_root("vocab_size"))
            .ok_or_else(|| anyhow::anyhow!("config missing vocab_size"))?
            as u32;

        // head_dim may be explicit; otherwise hidden / num_heads.
        let head_dim = pick("head_dim")
            .or_else(|| pick_root("head_dim"))
            .map(|x| x as u32)
            .unwrap_or(hidden_size / num_attention_heads);

        // Quantization lives at the root (applies to whole model).
        let (quant_group_size, quant_bits) = root
            .get("quantization")
            .and_then(|q| q.as_object())
            .map(|q| {
                (
                    q.get("group_size")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                    q.get("bits").and_then(|v| v.as_u64()).map(|v| v as u32),
                )
            })
            .unwrap_or((None, None));

        let mtp_num_hidden_layers = pick("mtp_num_hidden_layers")
            .or_else(|| pick_root("mtp_num_hidden_layers"))
            .unwrap_or(0) as u32;
        let num_nextn_predict_layers = pick("num_nextn_predict_layers")
            .or_else(|| pick_root("num_nextn_predict_layers"))
            .unwrap_or(0) as u32;
        let mtp = mtp_info_from_config(mtp_num_hidden_layers, num_nextn_predict_layers);

        let model_type = root
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Compute dtype probe order:
        //   1. Quantized models: mlx-lm stores scales as fp16 and
        //      dispatches `quantized_matmul` in fp16, regardless of
        //      what the original `torch_dtype` was. Qwen2.5-Coder-4bit
        //      announces `torch_dtype: bfloat16` (the original training
        //      precision) but its scales are fp16 and decode runs in
        //      fp16 — trust the runtime over the announcement.
        //   2. `torch_dtype` / `dtype` field at root or text_cfg.
        //   3. Default to bf16 (Qwen2.5-bf16's norm).
        let activation_dtype = if quant_bits.is_some() {
            Dtype::F16
        } else {
            let dtype_str = root
                .get("torch_dtype")
                .or_else(|| text_cfg.get("torch_dtype"))
                .or_else(|| root.get("dtype"))
                .and_then(|v| v.as_str());
            match dtype_str {
                Some("bfloat16") | Some("bf16") => Dtype::Bf16,
                Some("float16") | Some("fp16") | Some("half") => Dtype::F16,
                Some("float32") | Some("fp32") => Dtype::F32,
                _ => Dtype::Bf16,
            }
        };

        Ok(Self {
            model_type,
            hidden_size,
            num_hidden_layers,
            num_kv_shared_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            quant_group_size,
            quant_bits,
            mtp_num_hidden_layers,
            num_nextn_predict_layers,
            mtp,
            activation_dtype,
        })
    }
}

fn augment_missing_text_meta_from_safetensors(
    config_path: &std::path::Path,
    root: &mut serde_json::Value,
) {
    let Some(dir) = config_path.parent() else {
        return;
    };
    let Some(shapes) = infer_text_weight_shapes(dir) else {
        return;
    };

    let target = if root.get("text_config").is_some() {
        root.get_mut("text_config").and_then(|v| v.as_object_mut())
    } else {
        root.as_object_mut()
    };
    let Some(target) = target else {
        return;
    };

    if !target.contains_key("vocab_size") {
        if let Some(vocab) = shapes.vocab_size {
            target.insert("vocab_size".to_string(), serde_json::json!(vocab));
        }
    }
    if !target.contains_key("head_dim") {
        if let Some(head_dim) = shapes.head_dim {
            target.insert("head_dim".to_string(), serde_json::json!(head_dim));
        }
    }
    if !target.contains_key("num_attention_heads") {
        if let (Some(q_dim), Some(head_dim)) = (shapes.q_dim, shapes.head_dim) {
            if head_dim > 0 && q_dim % head_dim == 0 {
                target.insert(
                    "num_attention_heads".to_string(),
                    serde_json::json!(q_dim / head_dim),
                );
            }
        }
    }
    if !target.contains_key("num_key_value_heads") {
        if let (Some(k_dim), Some(head_dim)) = (shapes.k_dim, shapes.head_dim) {
            if head_dim > 0 && k_dim % head_dim == 0 {
                target.insert(
                    "num_key_value_heads".to_string(),
                    serde_json::json!(k_dim / head_dim),
                );
            }
        }
    }
}

#[derive(Default)]
struct TextWeightShapes {
    vocab_size: Option<u64>,
    q_dim: Option<u64>,
    k_dim: Option<u64>,
    head_dim: Option<u64>,
}

fn infer_text_weight_shapes(dir: &std::path::Path) -> Option<TextWeightShapes> {
    use std::io::Read;

    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with(".safetensors"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    let mut out = TextWeightShapes::default();
    for path in paths {
        let Ok(mut file) = std::fs::File::open(&path) else {
            continue;
        };
        let mut len_buf = [0u8; 8];
        if file.read_exact(&mut len_buf).is_err() {
            continue;
        }
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header_buf = vec![0u8; header_len];
        if file.read_exact(&mut header_buf).is_err() {
            continue;
        }
        let Ok(header) = serde_json::from_slice::<serde_json::Value>(&header_buf) else {
            continue;
        };
        let Some(obj) = header.as_object() else {
            continue;
        };
        for (key, value) in obj {
            if key == "__metadata__" || key.starts_with("vision_tower.") {
                continue;
            }
            let Some(shape) = value.get("shape").and_then(|v| v.as_array()) else {
                continue;
            };
            let dim0 = shape.first().and_then(|v| v.as_u64());
            if out.vocab_size.is_none() && key.ends_with("embed_tokens.weight") {
                out.vocab_size = dim0;
            } else if out.q_dim.is_none() && key.ends_with(".layers.0.self_attn.q_proj.weight") {
                out.q_dim = dim0;
            } else if out.k_dim.is_none() && key.ends_with(".layers.0.self_attn.k_proj.weight") {
                out.k_dim = dim0;
            } else if out.head_dim.is_none()
                && (key.ends_with(".layers.0.self_attn.q_norm.weight")
                    || key.ends_with(".layers.0.self_attn.k_norm.weight"))
            {
                out.head_dim = dim0;
            }
        }
    }

    (out.vocab_size.is_some()
        || (out.q_dim.is_some() && out.k_dim.is_some() && out.head_dim.is_some()))
    .then_some(out)
}

fn mtp_info_from_config(mtp_num_hidden_layers: u32, num_nextn_predict_layers: u32) -> MtpInfo {
    if num_nextn_predict_layers > 0 {
        return MtpInfo {
            format: MtpFormat::NextN,
            declared_layers: num_nextn_predict_layers,
            max_depth: Some(num_nextn_predict_layers),
            has_weights: false,
            source: "config.num_nextn_predict_layers".to_string(),
        };
    }
    if mtp_num_hidden_layers > 0 {
        return MtpInfo {
            format: MtpFormat::QwenNative,
            declared_layers: mtp_num_hidden_layers,
            max_depth: None,
            has_weights: false,
            source: "config.mtp_num_hidden_layers".to_string(),
        };
    }
    MtpInfo::none()
}

fn detect_mtp_info(
    config_path: &std::path::Path,
    root: &serde_json::Value,
    mtp_num_hidden_layers: u32,
    num_nextn_predict_layers: u32,
) -> MtpInfo {
    let Some(dir) = config_path.parent() else {
        return mtp_info_from_config(mtp_num_hidden_layers, num_nextn_predict_layers);
    };

    if let Some(info) =
        detect_mtplx_info(dir, root, mtp_num_hidden_layers, num_nextn_predict_layers)
    {
        return info;
    }

    let has_weights = has_mtp_weight_keys(dir);
    let mut info = mtp_info_from_config(mtp_num_hidden_layers, num_nextn_predict_layers);
    if info.is_declared() {
        info.has_weights = has_weights;
    }
    info
}

fn detect_mtplx_info(
    dir: &std::path::Path,
    root: &serde_json::Value,
    mtp_num_hidden_layers: u32,
    num_nextn_predict_layers: u32,
) -> Option<MtpInfo> {
    let path = dir.join("mtplx_runtime.json");
    if !path.is_file() {
        return None;
    }
    let value: serde_json::Value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())?;
    let object = value.as_object()?;
    let arch = object
        .get("arch_id")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    if !object.contains_key("mtplx_version") && !arch.to_ascii_lowercase().contains("mtp") {
        return None;
    }

    let sidecar = root
        .get("mlx_lm_extra_tensors")
        .and_then(|v| v.as_object())
        .and_then(|o| o.get("mtp_file"))
        .and_then(|v| v.as_str());
    let has_sidecar = sidecar
        .map(|name| dir.join(name).is_file())
        .unwrap_or_else(|| dir.join("mtp.safetensors").is_file());
    let declared_layers = num_nextn_predict_layers.max(mtp_num_hidden_layers);
    let max_depth = object
        .get("mtp_depth_max")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Some(MtpInfo {
        format: MtpFormat::Mtplx,
        declared_layers,
        max_depth,
        has_weights: has_sidecar || has_mtp_weight_keys(dir),
        source: "mtplx_runtime.json".to_string(),
    })
}

fn has_mtp_weight_keys(dir: &std::path::Path) -> bool {
    if has_mtp_weight_keys_in_index(&dir.join("model.safetensors.index.json")) {
        return true;
    }

    let mut paths: Vec<std::path::PathBuf> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.ends_with(".safetensors"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => return false,
    };
    paths.sort();
    paths
        .iter()
        .any(|path| has_mtp_weight_keys_in_safetensors(path))
}

fn has_mtp_weight_keys_in_index(path: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    value
        .get("weight_map")
        .and_then(|v| v.as_object())
        .map(|map| map.keys().any(|key| is_mtp_weight_key(key)))
        .unwrap_or(false)
}

fn has_mtp_weight_keys_in_safetensors(path: &std::path::Path) -> bool {
    use std::io::Read;

    let Ok(mut file) = std::fs::File::open(path) else {
        return false;
    };
    let mut len_buf = [0u8; 8];
    if file.read_exact(&mut len_buf).is_err() {
        return false;
    }
    let header_len = u64::from_le_bytes(len_buf) as usize;
    if header_len > 128 * 1024 * 1024 {
        return false;
    }
    let mut header = vec![0u8; header_len];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&header) else {
        return false;
    };
    value
        .as_object()
        .map(|object| object.keys().any(|key| is_mtp_weight_key(key)))
        .unwrap_or(false)
}

fn is_mtp_weight_key(key: &str) -> bool {
    key.starts_with("mtp.") || key.contains(".mtp.")
}

fn infer_quant_activation_dtype_from_safetensors(config_path: &std::path::Path) -> Option<Dtype> {
    use std::io::Read;

    let dir = config_path.parent()?;
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with(".safetensors"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();

    for path in &paths {
        let mut file = std::fs::File::open(path).ok()?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf).ok()?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        if header_len > 128 * 1024 * 1024 {
            return None;
        }
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header).ok()?;
        let json: serde_json::Value = serde_json::from_slice(&header).ok()?;
        let object = json.as_object()?;

        for (name, tensor) in object {
            if name == "__metadata__" {
                continue;
            }
            if !name.ends_with(".scales") && !name.ends_with("scales") {
                continue;
            }
            if let Some(dtype) = tensor
                .get("dtype")
                .and_then(|v| v.as_str())
                .and_then(dtype_from_safetensors_label)
            {
                return Some(dtype);
            }
        }
    }

    None
}

fn dtype_from_safetensors_label(label: &str) -> Option<Dtype> {
    match label {
        "BF16" => Some(Dtype::Bf16),
        "F16" => Some(Dtype::F16),
        "F32" => Some(Dtype::F32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Write;

    fn base_config(extra: serde_json::Value) -> serde_json::Value {
        let mut cfg = json!({
            "model_type": "qwen3_5",
            "hidden_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 256,
            "vocab_size": 1024
        });
        let object = cfg.as_object_mut().unwrap();
        for (key, value) in extra.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        cfg
    }

    fn write_json(path: &std::path::Path, value: &serde_json::Value) {
        std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
    }

    fn write_fake_safetensors(path: &std::path::Path, header: serde_json::Value) {
        let bytes = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
    }

    #[test]
    fn mtplx_sidecar_supplies_explicit_depth() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(
            &tmp.path().join("config.json"),
            &base_config(json!({
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "text_config": {
                    "hidden_size": 128,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "num_hidden_layers": 2,
                    "intermediate_size": 256,
                    "vocab_size": 1024,
                    "mtp_num_hidden_layers": 1
                }
            })),
        );
        write_json(
            &tmp.path().join("mtplx_runtime.json"),
            &json!({"mtplx_version": "0.1.0-preview", "mtp_depth_max": 2}),
        );
        std::fs::write(tmp.path().join("mtp.safetensors"), []).unwrap();

        let meta = ModelMeta::from_config_file(&tmp.path().join("config.json")).unwrap();
        assert_eq!(meta.mtp.format, MtpFormat::Mtplx);
        assert_eq!(meta.mtp.max_depth, Some(2));
        assert!(meta.mtp.has_weights);
    }

    #[test]
    fn nextn_config_uses_declared_depth_when_weights_exist() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(
            &tmp.path().join("config.json"),
            &base_config(json!({"num_nextn_predict_layers": 2})),
        );
        write_json(
            &tmp.path().join("model.safetensors.index.json"),
            &json!({"weight_map": {"mtp.0.block.ffn.weight": "model-00001.safetensors"}}),
        );

        let meta = ModelMeta::from_config_file(&tmp.path().join("config.json")).unwrap();
        assert_eq!(meta.mtp.format, MtpFormat::NextN);
        assert_eq!(meta.mtp.max_depth, Some(2));
        assert!(meta.mtp.has_weights);
    }

    #[test]
    fn qwen_native_mtp_requires_weights_for_activation() {
        let tmp = tempfile::tempdir().unwrap();
        write_json(
            &tmp.path().join("config.json"),
            &base_config(json!({"mtp_num_hidden_layers": 1})),
        );

        let meta = ModelMeta::from_config_file(&tmp.path().join("config.json")).unwrap();
        assert_eq!(meta.mtp.format, MtpFormat::QwenNative);
        assert_eq!(meta.mtp.max_depth, None);
        assert!(!meta.mtp.has_weights);

        write_fake_safetensors(
            &tmp.path().join("model.safetensors"),
            json!({
                "__metadata__": {},
                "language_model.mtp.layers.0.mlp.gate_proj.weight": {
                    "dtype": "F16",
                    "shape": [1],
                    "data_offsets": [0, 2]
                }
            }),
        );
        let meta = ModelMeta::from_config_file(&tmp.path().join("config.json")).unwrap();
        assert!(meta.mtp.has_weights);
    }
}
