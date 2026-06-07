//! Top-level IR container.
//!
//! A `Module` is the unit of compilation — captures the model's
//! forward graph after lowering from `fusion::FusionGraph`. Passes
//! mutate the module; the final lowering step emits MSL from it.

use std::collections::BTreeMap;

use crate::graph::{Dtype, LayerTag, TensorId};
use crate::op::op::Op;

/// IR-level tensor metadata. Mirrors `graph::Tensor` but adds fields
/// the passes need: strides (set by layout inference, step 2),
/// partitioning (set by scheduler, step 5).
///
/// Naming: `op::Tensor` (IR-level, this type) vs `graph::Tensor`
/// (raw mlx trace tensor) — distinguished by module path.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Vec<i32>,
    pub dtype: Dtype,

    /// Producer `Op` index (`None` for module inputs / leaf params).
    pub producer: Option<u32>,

    /// Stride per dim, in elements. `None` until layout inference
    /// (IR step 2) runs. Default semantics: row-major contiguous
    /// (stride[i] = product of shape[i+1..]).
    pub strides: Option<Vec<i64>>,

    /// Which axes are sharded across threadgroups / simdgroups /
    /// lanes. `None` until the scheduler (step 5) assigns them.
    pub partitioning: Option<Vec<Partition>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Partition {
    /// This axis is iterated by each lane (sequential).
    Lane,
    /// This axis is sharded across simdgroups in a threadgroup.
    Simdgroup,
    /// This axis is sharded across threadgroups in the grid.
    Threadgroup,
    /// This axis is fully replicated (every lane sees every element).
    Replicated,
}

/// Top-level container.
#[derive(Debug, Clone)]
pub struct Module {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    /// Output tensors of the module (final logits, KV cache writes,
    /// next-token output). The DCE pass anchors backward reachability
    /// here.
    pub outputs: Vec<TensorId>,
    /// Free-form module-level metadata. Holds e.g. `hidden_size`,
    /// `num_attention_heads`, `head_dim` so passes can read model
    /// config without going through `ModelMeta` directly.
    pub meta: BTreeMap<String, ModuleMetaValue>,
    /// Which passes have already run on this module (for diagnostics
    /// and idempotency checks).
    pub passes_applied: Vec<String>,
}

impl Module {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            ops: Vec::new(),
            outputs: Vec::new(),
            meta: BTreeMap::new(),
            passes_applied: Vec::new(),
        }
    }

    /// Iterate the ops attached to a layer. Pass hooks use this for
    /// layer-local rewrites.
    pub fn ops_for_layer(&self, layer: LayerTag) -> impl Iterator<Item = &Op> {
        self.ops
            .iter()
            .filter(move |op| op.layer_hint == Some(layer))
    }

    pub fn tensor(&self, tid: TensorId) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.id == tid)
    }

    pub fn op(&self, id: u32) -> Option<&Op> {
        self.ops.iter().find(|op| op.id == id)
    }

    /// Mutable lookup — used by passes that rewrite tensor metadata
    /// (layout inference, scheduler).
    pub fn tensor_mut(&mut self, tid: TensorId) -> Option<&mut Tensor> {
        self.tensors.iter_mut().find(|t| t.id == tid)
    }
}

impl Tensor {
    /// Row-major contiguous strides for a given shape. The trailing
    /// dim has stride 1; each preceding dim has stride =
    /// `product(shape[i+1..])`. This is the default mlx layout for
    /// freshly-materialised tensors. Reshape / Transpose / Broadcast
    /// inference rewrites these.
    pub fn row_major_strides(shape: &[i32]) -> Vec<i64> {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * (shape[i + 1].max(0) as i64);
        }
        strides
    }

    /// Test whether this tensor's strides match row-major contiguous
    /// layout. `true` for both un-inferred tensors (no strides set
    /// yet — treat as default contiguous) and tensors whose strides
    /// were explicitly inferred to match the canonical layout. A view
    /// op (Reshape that doesn't permute, Squeeze, ExpandDims,
    /// AsContiguous) preserves this property; Transpose / Broadcast
    /// generally break it.
    pub fn is_row_major_contiguous(&self) -> bool {
        let expected = Self::row_major_strides(&self.shape);
        match &self.strides {
            None => true,
            Some(s) => *s == expected,
        }
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum ModuleMetaValue {
    I32(i32),
    U32(u32),
    F32(f32),
    String(String),
}
