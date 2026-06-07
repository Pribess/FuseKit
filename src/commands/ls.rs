use anyhow::Result;
use comfy_table::{
    modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Attribute, Cell, CellAlignment, Color,
    ContentArrangement, Table,
};
use owo_colors::OwoColorize;

use crate::fusion::ir::MtpFormat;
use crate::omlx;

pub fn run() -> Result<()> {
    let dir = omlx::models_dir()?;
    let models = omlx::list_models()?;

    println!(
        "{} {}",
        "oMLX models".bold(),
        format!("({})", dir.display()).dimmed()
    );

    if models.is_empty() {
        println!("  {}", "(no models found)".dimmed());
        return Ok(());
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Status").add_attribute(Attribute::Bold),
            Cell::new("Model").add_attribute(Attribute::Bold),
            Cell::new("Type").add_attribute(Attribute::Bold),
            Cell::new("MTP").add_attribute(Attribute::Bold),
            Cell::new("Size")
                .add_attribute(Attribute::Bold)
                .set_alignment(CellAlignment::Right),
        ]);

    for m in &models {
        let (status_cell, name_color) = if m.is_patched() {
            (Cell::new("● patched").fg(Color::Green), Color::Green)
        } else {
            (Cell::new("○ none").fg(Color::DarkGrey), Color::Reset)
        };

        let model_type = m.model_type().unwrap_or_else(|| "?".to_string());
        let mtp = match m.mtp_info() {
            Some(info) if info.format == MtpFormat::Mtplx => {
                format!(
                    "mtplx:{}",
                    info.max_depth
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "?".to_string())
                )
            }
            Some(info) if info.format == MtpFormat::NextN => {
                format!("nextn:{}", info.declared_layers)
            }
            Some(info) if info.format == MtpFormat::QwenNative => {
                let suffix = if info.has_weights { "" } else { " no-weights" };
                format!("qwen:{}{}", info.declared_layers, suffix)
            }
            _ => "—".to_string(),
        };
        let size = omlx::format_size(m.weight_bytes());

        table.add_row(vec![
            status_cell,
            Cell::new(&m.name).fg(name_color),
            Cell::new(model_type).fg(Color::Cyan),
            Cell::new(mtp).fg(Color::Magenta),
            Cell::new(size).set_alignment(CellAlignment::Right),
        ]);
    }

    println!("{table}");

    let total = models.len();
    let patched = models.iter().filter(|m| m.is_patched()).count();
    let summary = format!(
        "{} model(s), {} patched",
        total,
        if patched > 0 {
            patched.to_string().green().to_string()
        } else {
            patched.to_string().dimmed().to_string()
        }
    );
    println!("{}", summary);

    Ok(())
}
