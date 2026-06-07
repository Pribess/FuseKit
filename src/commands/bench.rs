//! `fusekit bench` — fixed-length mlx-lm/fusekit benchmark.

use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use std::process::Command;

use crate::{omlx, paths, ui};

pub struct BenchOptions<'a> {
    pub model: &'a str,
    pub tokens: usize,
    pub prompt: &'a str,
    pub repeat: usize,
    pub max_kv_size: Option<usize>,
    pub kv_bits: Option<u8>,
    pub kv_group_size: usize,
    pub raw: bool,
    pub per_kernel: bool,
    pub dump_tokens: bool,
}

pub fn run(opts: BenchOptions<'_>) -> Result<()> {
    if opts.raw && opts.per_kernel {
        return Err(anyhow!("--raw and --per-kernel are mutually exclusive"));
    }
    if let Some(bits) = opts.kv_bits {
        if bits != 4 && bits != 8 {
            return Err(anyhow!("--kv-bits must be 4 or 8"));
        }
    }

    let m = omlx::find_model(opts.model)?;
    let python = paths::mlx_python_bin()?;
    let site_packages = paths::mlx_site_packages()?;
    let script = locate_bench_script()?;

    ui::ok(format!("model: {} ({})", m.name, m.path.display()));
    if opts.raw {
        ui::dim_kv("mode", "mlx-lm raw (FUSEKIT_DISABLE_HOOK=1)");
    } else if opts.per_kernel {
        ui::dim_kv("mode", "fusekit general + per-kernel profile");
    } else {
        ui::dim_kv("mode", "fusekit general");
    }

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg(&m.path)
        .args(["--tokens", &opts.tokens.to_string()])
        .args(["--prompt", opts.prompt])
        .args(["--repeat", &opts.repeat.to_string()])
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONPATH", &site_packages);

    if let Some(max_kv_size) = opts.max_kv_size {
        cmd.args(["--max-kv-size", &max_kv_size.to_string()]);
    }
    if let Some(kv_bits) = opts.kv_bits {
        cmd.args(["--kv-bits", &kv_bits.to_string()]);
        cmd.args(["--kv-group-size", &opts.kv_group_size.to_string()]);
    }
    if opts.dump_tokens {
        cmd.arg("--dump-tokens");
    }
    if opts.raw {
        cmd.env("FUSEKIT_DISABLE_HOOK", "1");
    }
    if opts.per_kernel {
        cmd.env("FUSEKIT_PROFILE_KERNELS", "1");
    }

    let status = cmd.status().context("running bench.py")?;
    if !status.success() {
        return Err(anyhow!("bench.py exited with {status}"));
    }
    Ok(())
}

fn locate_bench_script() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let candidates = [
        exe.parent().map(|p| p.join("scripts/bench.py")),
        exe.parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("scripts/bench.py")),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/bench.py")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.is_file() {
            return Ok(c);
        }
    }
    Err(anyhow!("bench.py not found"))
}
