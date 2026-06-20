//! Pass infrastructure + concrete passes.
//!
//! Passes implement `Pass` and are sequenced through `PassManager`.
//! Submodules host individual passes; this file holds the trait,
//! manager, and shared error type. Add new passes by creating a file
//! in `passes/`, implementing `Pass`, and re-exporting below.

use std::fmt;

use crate::op::module::Module;

pub mod algebraic;
pub mod cost;
pub mod dce;
pub mod decode_specialize;
pub mod inplace;
pub mod layout_inference;
pub mod macros;

pub use algebraic::AlgebraicRewrites;
pub use dce::DeadCodeElimination;
pub use decode_specialize::DecodeL1Specialize;
pub use inplace::InPlaceAnalysis;
pub use layout_inference::LayoutInference;
pub use macros::{KvSliceUpdateContiguous, RmsNormFastRsqrt, SoftmaxMaxStable};

#[derive(Debug)]
pub enum PassError {
    /// Pass detected an invariant violation in the IR.
    Invariant(String),
    /// Pass couldn't apply to this module (e.g. unsupported op).
    Unsupported(String),
}

impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PassError::Invariant(msg) => write!(f, "invariant violation: {msg}"),
            PassError::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for PassError {}

pub trait Pass {
    /// Short name for diagnostics + idempotency tracking.
    fn name(&self) -> &'static str;

    /// Transform `module` in place. Returns `true` if anything changed
    /// (drives fixed-point loops where caller iterates).
    fn run(&self, module: &mut Module) -> Result<bool, PassError>;
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run all registered passes once over the module in registration
    /// order. Records each pass's name on `module.passes_applied`.
    pub fn run(&self, module: &mut Module) -> Result<(), PassError> {
        for pass in &self.passes {
            pass.run(module)?;
            module.passes_applied.push(pass.name().to_string());
        }
        Ok(())
    }

    /// Run passes to fixed point — keep iterating until no pass
    /// reports a change. Caps at `max_iters` to bound compile time.
    pub fn run_to_fixed_point(
        &self,
        module: &mut Module,
        max_iters: usize,
    ) -> Result<(), PassError> {
        for _ in 0..max_iters {
            let mut changed = false;
            for pass in &self.passes {
                if pass.run(module)? {
                    changed = true;
                    module.passes_applied.push(pass.name().to_string());
                }
            }
            if !changed {
                return Ok(());
            }
        }
        Err(PassError::Invariant(format!(
            "pass manager hit {max_iters}-iter cap without converging"
        )))
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

/// No-op pass — useful as a placeholder while the pass infrastructure
/// is built out, and for tests.
pub struct NoOpPass;

impl Pass for NoOpPass {
    fn name(&self) -> &'static str {
        "noop"
    }

    fn run(&self, _module: &mut Module) -> Result<bool, PassError> {
        Ok(false)
    }
}
