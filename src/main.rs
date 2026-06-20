use clap::{Parser, Subcommand};

mod codegen;
mod commands;
mod fusion;
mod graph;
mod hook;
mod layer_tag;
mod loader;
mod manifest;
mod omlx;
mod op;
mod paths;
mod profile;
mod runtime;
mod runtime_payload;
mod target;
mod task;
mod ui;

#[derive(Parser)]
#[command(name = "fusekit")]
#[command(version)]
#[command(about = "Apple Metal megakernel compiler for mlx-lm models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List oMLX models with patch status
    Ls,

    /// Compile megakernel for a model and install mlx-lm hook
    Patch {
        /// Model directory name (under oMLX models dir)
        model: String,
        /// Target chip for codegen budgets (registers, SMEM, TG sizing).
        /// `auto` (default) detects the host; explicit values: m1..m5.
        #[arg(long, value_name = "CHIP")]
        chip: Option<String>,
        /// Active token rows to emit for dense/routing kernels.
        /// Default decode mode emits M=1; prefill/MTP experiments can
        /// request M>1 and the value is clamped to the captured trace M.
        #[arg(long, value_name = "N")]
        active_m: Option<u32>,
        /// Dense matmul codegen mode: decode-gemv | simdgroup-gemm | auto.
        /// `auto` enables simdgroup matrix kernels once active M is large
        /// enough for GEMM-style tiling.
        #[arg(long, value_name = "MODE")]
        dense_matmul: Option<String>,
        /// Rebuild the model's fusekit artifacts even if the manifest
        /// already matches this binary.
        #[arg(long)]
        force: bool,
        /// Hand-tuned pattern matchers: `auto` (default) routes through
        /// the compiler-path general fusion (the pattern fast-path is
        /// currently slower than current mlx-lm, so default delivers
        /// mlx-equivalent speed via the fallback), `on` keeps the
        /// legacy pattern path enabled (useful for measuring against
        /// the hand-tuned baseline), `off` forces compiler-only.
        #[arg(long, value_name = "AUTO|ON|OFF", default_value = "auto")]
        patterns: String,
    },

    /// Remove compiled megakernel; auto-remove mlx-lm hook if last patched model
    Unpatch { model: String },

    /// Show overall fusekit status
    Status,

    /// Run a fixed-length benchmark through raw mlx-lm or fusekit
    Bench {
        /// Model directory name (under oMLX models dir)
        model: String,
        /// Number of generated tokens
        #[arg(long, default_value_t = 64)]
        tokens: usize,
        /// Prompt to feed the model
        #[arg(
            long,
            default_value = "Repeat the word token forever: token token token"
        )]
        prompt: String,
        /// Number of benchmark repetitions
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// Optional max KV cache size passed to mlx-lm
        #[arg(long)]
        max_kv_size: Option<usize>,
        /// Use mlx-lm QuantizedKVCache with 4-bit or 8-bit entries
        #[arg(long, value_name = "4|8")]
        kv_bits: Option<u8>,
        /// Group size for QuantizedKVCache
        #[arg(long, default_value_t = 64)]
        kv_group_size: usize,
        /// Disable fusekit and measure mlx-lm raw
        #[arg(long)]
        raw: bool,
        /// Print per-kernel GPU timing for the first fusekit decode step
        #[arg(long)]
        per_kernel: bool,
        /// Print generated token ids
        #[arg(long)]
        dump_tokens: bool,
    },

    /// Capture a model's forward graph and print a summary
    Inspect {
        /// Model directory name (under the oMLX models dir)
        model: String,
    },

    /// Compile the megakernel pipeline end-to-end (dry run, no mlx-lm hook)
    Compile {
        /// Model directory name (under the oMLX models dir)
        model: String,
        /// Output directory (default: <model_dir>/fusekit)
        #[arg(long)]
        out: Option<std::path::PathBuf>,
        /// Target chip for codegen budgets (registers, SMEM, TG sizing).
        /// `auto` (default) detects the host; explicit values: m1..m5.
        #[arg(long, value_name = "CHIP")]
        chip: Option<String>,
        /// Active token rows to emit for dense/routing kernels.
        #[arg(long, value_name = "N")]
        active_m: Option<u32>,
        /// Dense matmul codegen mode: decode-gemv | simdgroup-gemm | auto.
        #[arg(long, value_name = "MODE")]
        dense_matmul: Option<String>,
        /// Hand-tuned fusion patterns: `auto` (default) | `on` | `off`.
        /// `auto`/`off` route through compiler-path general fusion;
        /// `on` keeps legacy pattern matchers enabled for comparison.
        #[arg(long, value_name = "AUTO|ON|OFF", default_value = "auto")]
        patterns: String,
    },

    /// Record a unified CPU+GPU trace via xctrace
    Profile {
        /// Model directory name (under the oMLX models dir)
        model: String,
        /// Where to save the .trace file (default: /tmp/fusekit-<timestamp>.trace)
        #[arg(long)]
        output: Option<std::path::PathBuf>,
        /// Open the resulting trace in Instruments when done
        #[arg(long)]
        open: bool,
        /// Prompt to feed the model
        #[arg(long, default_value = "Write a short poem about the ocean.")]
        prompt: String,
        /// Number of tokens to decode
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
    },

    /// Inner subcommand invoked by xctrace — not for direct use
    #[command(hide = true)]
    ProfileInner {
        model: String,
        #[arg(long)]
        prompt: String,
        #[arg(long)]
        max_tokens: usize,
    },

    /// Probe the embedded Python interpreter (hidden)
    #[command(hide = true)]
    PyProbe,

    /// Smoke test the Metal runtime: dispatch elementwise_add and verify
    #[command(hide = true)]
    Smoke,

    /// Verify a compiled task_graph.json + metallib resolves against
    /// the live model, and optionally execute the full dispatch list
    /// once as a smoke test (`--execute`).
    #[command(hide = true)]
    VerifyPlan {
        model: String,
        /// Run every task once via the runtime dispatcher. Outputs are
        /// not meaningful — this is a wiring smoke check.
        #[arg(long)]
        execute: bool,
    },
}

fn apply_codegen_knobs(active_m: Option<u32>, dense_matmul: Option<&str>) -> anyhow::Result<()> {
    if let Some(n) = active_m {
        if n == 0 {
            anyhow::bail!("--active-m must be >= 1");
        }
        std::env::set_var("FUSEKIT_ACTIVE_M", n.to_string());
    }
    if let Some(raw) = dense_matmul {
        let mode = match raw.trim().to_ascii_lowercase().as_str() {
            "decode-gemv" | "gemv" | "decode" => "decode-gemv",
            "simdgroup-gemm" | "gemm" | "simdgroup" => "simdgroup-gemm",
            "auto" => "auto",
            other => {
                anyhow::bail!(
                    "--dense-matmul must be one of: decode-gemv | simdgroup-gemm | auto (got {other})"
                );
            }
        };
        std::env::set_var("FUSEKIT_DENSE_MATMUL_MODE", mode);
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Ls => commands::ls::run(),
        Commands::Patch {
            model,
            chip,
            active_m,
            dense_matmul,
            force,
            patterns,
        } => {
            apply_codegen_knobs(active_m, dense_matmul.as_deref())?;
            // `auto` (new default) / `off` route through compiler-path
            // general fusion. The pattern fast-path is currently
            // slower than the post-update mlx-lm and is left as an
            // opt-in fallback for measurement work. Compiler-path also
            // produces strictly more chain coverage (241 vs 48 on
            // 0.5B-4bit) so future kernel improvements compound there.
            match patterns.to_ascii_lowercase().as_str() {
                "auto" | "off" => {
                    std::env::set_var("FUSEKIT_NO_PATTERNS", "1");
                    std::env::set_var("FUSEKIT_GENERAL_FUSION", "1");
                }
                "on" => {}
                other => {
                    anyhow::bail!("--patterns must be one of: auto | on | off (got {other})");
                }
            }
            commands::patch::run(&model, chip.as_deref(), force)
        }
        Commands::Unpatch { model } => commands::unpatch::run(&model),
        Commands::Status => commands::status::run(),
        Commands::Bench {
            model,
            tokens,
            prompt,
            repeat,
            max_kv_size,
            kv_bits,
            kv_group_size,
            raw,
            per_kernel,
            dump_tokens,
        } => commands::bench::run(commands::bench::BenchOptions {
            model: &model,
            tokens,
            prompt: &prompt,
            repeat,
            max_kv_size,
            kv_bits,
            kv_group_size,
            raw,
            per_kernel,
            dump_tokens,
        }),
        Commands::Inspect { model } => commands::inspect::run(&model),
        Commands::Compile {
            model,
            out,
            chip,
            active_m,
            dense_matmul,
            patterns,
        } => {
            apply_codegen_knobs(active_m, dense_matmul.as_deref())?;
            // Keep compile and patch semantics aligned: pattern matchers
            // are opt-in, while the default compiler path is general
            // fusion.
            match patterns.to_ascii_lowercase().as_str() {
                "auto" | "off" => {
                    std::env::set_var("FUSEKIT_NO_PATTERNS", "1");
                    std::env::set_var("FUSEKIT_GENERAL_FUSION", "1");
                }
                "on" => {}
                other => {
                    anyhow::bail!("--patterns must be one of: auto | on | off (got {other})");
                }
            }
            commands::compile::run(&model, out, chip.as_deref())
        }
        Commands::Profile {
            model,
            output,
            open,
            prompt,
            max_tokens,
        } => commands::profile::run(&model, output, open, &prompt, max_tokens),
        Commands::ProfileInner {
            model,
            prompt,
            max_tokens,
        } => commands::profile::run_inner(&model, &prompt, max_tokens),
        Commands::PyProbe => commands::py_probe::run(),
        Commands::Smoke => commands::smoke::run(),
        Commands::VerifyPlan { model, execute } => commands::verify_plan::run(&model, execute),
    }
}
