//! Small UI helpers shared across commands.
//!
//! Owo-colors auto-disables on non-TTY stdout, so these prints stay safe in
//! pipes and log captures.

use owo_colors::OwoColorize;

pub fn ok(msg: impl AsRef<str>) {
    println!("{} {}", "✓".green().bold(), msg.as_ref());
}

pub fn skip(msg: impl AsRef<str>) {
    println!("{} {}", "·".dimmed(), msg.as_ref().dimmed());
}

pub fn step(msg: impl AsRef<str>) {
    println!("{} {}", "→".cyan().bold(), msg.as_ref());
}

pub fn dim_kv(key: &str, val: impl std::fmt::Display) {
    println!("  {} {}", format!("{key}:").dimmed(), val);
}
