use anyhow::Result;
use owo_colors::OwoColorize;

use crate::{hook, omlx};

pub fn run() -> Result<()> {
    let label_w = 18;

    print_row(
        label_w,
        "oMLX models dir",
        match omlx::models_dir() {
            Ok(dir) => format!("{}", dir.display().green()),
            Err(e) => format!("{} {}", "not found".red(), format!("({e})").dimmed()),
        },
    );

    print_row(
        label_w,
        "mlx-lm hook",
        match hook::status() {
            Ok((path, hook::HookStatus::Installed)) => format!(
                "{}  {}",
                "● installed".green(),
                path.display().to_string().dimmed()
            ),
            Ok((path, hook::HookStatus::NotInstalled)) => format!(
                "{}  {}",
                "○ not installed".yellow(),
                path.display().to_string().dimmed()
            ),
            Err(e) => format!("{} {}", "unavailable".red(), format!("({e})").dimmed()),
        },
    );

    if let Ok(models) = omlx::list_models() {
        let total = models.len();
        let patched = models.iter().filter(|m| m.is_patched()).count();
        let total_size: u64 = models.iter().map(|m| m.weight_bytes()).sum();
        let summary = format!(
            "{} / {} patched   {}",
            if patched > 0 {
                patched.to_string().green().bold().to_string()
            } else {
                patched.to_string().dimmed().to_string()
            },
            total.to_string().bold(),
            format!("({} total)", omlx::format_size(total_size)).dimmed()
        );
        print_row(label_w, "models", summary);
    }

    Ok(())
}

fn print_row(label_w: usize, label: &str, value: String) {
    println!("{:<label_w$} {}", format!("{label}:").bold(), value);
}
