//! `fusekit smoke` — sanity-check the Metal runtime end to end.
//!
//! Builds two host-side bf16 arrays, dispatches the bundled
//! `elementwise_add_bf16` kernel, and verifies the GPU result against
//! a CPU reference. If this passes, our metallib → Metal pipeline →
//! dispatch → readback path is intact.

use anyhow::{anyhow, Result};
use half::bf16;

use crate::runtime::dispatch::elementwise_add_bf16;
use crate::runtime::metal::MetalContext;
use crate::ui;

pub fn run() -> Result<()> {
    let ctx = MetalContext::new()?;
    ui::ok(format!("metal device: {}", ctx.device.name()));

    let n: usize = 1024;
    // Keep magnitudes small — bf16 has 7-bit mantissa, so rounding
    // error grows with magnitude (≈ value × 2⁻⁷). Below 64 we have
    // precision around 0.5; the tolerance below assumes this range.
    let a: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32((i % 64) as f32 * 0.5))
        .collect();
    let b: Vec<bf16> = (0..n)
        .map(|i| bf16::from_f32((i % 64) as f32 * 0.25 + 1.0))
        .collect();

    let out = elementwise_add_bf16(&ctx, &a, &b)?;
    ui::ok(format!("dispatched, got {} elements", out.len()));

    // Compare against CPU reference. bf16 has ~7-bit mantissa precision,
    // so allow a small absolute tolerance.
    let mut max_err = 0.0f32;
    for i in 0..n {
        let expected = a[i].to_f32() + b[i].to_f32();
        let got = out[i].to_f32();
        let err = (expected - got).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // bf16 quantization at this magnitude range allows ~0.5 of slop.
    // We only care that the kernel runs, not that bf16 matches fp32.
    let tol = 0.5;
    if max_err > tol {
        return Err(anyhow!(
            "elementwise_add_bf16 max abs error {max_err} exceeds tolerance {tol}"
        ));
    }
    ui::ok(format!("max abs error: {max_err:.6} (tol {tol})"));
    Ok(())
}
