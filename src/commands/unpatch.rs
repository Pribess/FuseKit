use anyhow::{Context, Result};

use crate::{hook, omlx, ui};

pub fn run(model: &str) -> Result<()> {
    let m = omlx::find_model(model)?;
    let fusekit_dir = m.path.join("fusekit");

    if !fusekit_dir.is_dir() {
        ui::skip(format!("{model} is not patched"));
        return Ok(());
    }

    // The runtime lives inside fusekit_dir/runtime/, so removing the
    // directory wipes everything for this model in one shot.
    std::fs::remove_dir_all(&fusekit_dir)
        .with_context(|| format!("remove {}", fusekit_dir.display()))?;
    ui::ok(format!("unpatched {model}"));

    let any_remaining = omlx::list_models()?.iter().any(|m| m.is_patched());
    if !any_remaining {
        hook::uninstall().context("remove fusekit hook")?;
        ui::ok("hook removed (no patched models remain)");
    }
    Ok(())
}
