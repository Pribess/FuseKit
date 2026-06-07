//! Generic operation representation.
//!
//! Replaces `fusion::FusionKind`'s hardcoded variants
//! (`RmsNormQkv`, `LinearResidual`, etc.) with a single
//! `Op { kind, attrs }` shape. Pattern-recognised compositions become
//! `OpKind::Chain { steps }`; synthetic kernels (RoPE, embed lookup,
//! argmax) become `OpKind::Synthetic { name }` opaque blocks for now,
//! slated for decomposition in later passes.
//!
//! Naming: this `Op` is the IR-level fused op. The raw mlx trace op
//! lives at `crate::graph::Op` — module path disambiguates.

use std::collections::BTreeMap;

use crate::graph::{Attrs, Dtype, LayerTag, OpId, TensorId};

/// One node in the IR. Either a primitive op (matches an mlx trace
/// op), a fused chain (multiple primitives composed register-resident),
/// or an opaque "synthetic" decode kernel that hasn't been
/// decomposed yet.
#[derive(Debug, Clone)]
pub struct Op {
    pub id: u32,
    pub kind: OpKind,
    /// op_ids in the underlying `graph::Graph` this Op absorbed. Empty
    /// for `Synthetic` ops (no trace correspondence).
    pub op_ids: Vec<OpId>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub attrs: BTreeMap<String, AttrValue>,
    pub layer_hint: Option<LayerTag>,
}

/// What this IR node represents semantically.
#[derive(Debug, Clone)]
pub enum OpKind {
    /// A single mlx primitive (matmul, rmsnorm, softmax, add, ...).
    /// `name` is the mlx primitive name. `attrs` (on the parent `Op`)
    /// carries the per-primitive parameters (eps for rmsnorm, scale
    /// for sdpa, etc.).
    Primitive { name: String, raw_attrs: Attrs },

    /// A chain of primitives composed register-resident. Each step's
    /// inputs may reference the chain's external buffers
    /// (`Attr::External(idx)`) or an earlier step's register
    /// (`Attr::Register(step_idx)`).
    Chain { steps: Vec<ChainStep> },

    /// Synthetic decode-time kernel that doesn't correspond to mlx
    /// primitives (RoPE-Q, KV cache store, embed lookup, argmax). The
    /// IR keeps them opaque for now — passes are scheduled to
    /// decompose them into primitive chains once we land the
    /// algorithmic macros pass.
    Synthetic { name: String },
}

#[derive(Debug, Clone)]
pub struct ChainStep {
    pub op_name: String,
    pub raw_attrs: Attrs,
    /// Each operand: either reads an external buffer (by index into
    /// the chain's external inputs) or a register held by a previous
    /// step.
    pub inputs: Vec<Attr>,
    pub output_shape: Vec<i32>,
    pub output_dtype: Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attr {
    External { buffer_index: u32 },
    Register { step_index: u32 },
}

/// Attribute values associated with an `Op` (extensible map). Replaces
/// `FusionAttrs`'s closed enum so any architecture's op can carry its
/// own metadata without growing a central enum.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    I32(i32),
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
    Shape(Vec<i32>),
}

impl AttrValue {
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            AttrValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            AttrValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            AttrValue::U32(v) => Some(*v),
            _ => None,
        }
    }
}
