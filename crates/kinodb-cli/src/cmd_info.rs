use kinodb_core::KdbReader;

pub fn run(path: &str, show_episodes: bool) -> Result<(), Box<dyn std::error::Error>> {
    let reader = KdbReader::open(path)?;
    let header = reader.header();

    // ── File summary ────────────────────────────────────────
    println!("kinodb v{}.{}", header.version_major, header.version_minor);
    println!();
    println!("  File:      {}", path);
    println!("  Episodes:  {}", header.num_episodes);
    println!("  Frames:    {}", header.num_frames);

    if header.num_episodes > 0 {
        let avg_frames = header.num_frames as f64 / header.num_episodes as f64;
        println!("  Avg len:   {:.1} frames/episode", avg_frames);
    }

    // Compute file size
    let file_size = std::fs::metadata(path)?.len();
    println!("  File size: {}", format_bytes(file_size));
    println!(
        "  Created:   {}",
        format_timestamp(header.created_timestamp)
    );

    // ── Collect stats from episode metadata ──────────────────
    if reader.num_episodes() > 0 {
        let mut embodiments = std::collections::BTreeSet::new();
        let mut tasks = std::collections::BTreeSet::new();
        let mut success_count: u64 = 0;
        let mut success_known: u64 = 0;

        for i in 0..reader.num_episodes() {
            if let Ok(meta) = reader.read_meta(i) {
                embodiments.insert(meta.embodiment.clone());
                tasks.insert(meta.language_instruction.clone());
                if let Some(s) = meta.success {
                    success_known += 1;
                    if s {
                        success_count += 1;
                    }
                }
            }
        }

        println!();
        println!("  Embodiments ({}):", embodiments.len());
        for e in &embodiments {
            println!("    - {}", e);
        }

        println!();
        println!("  Tasks ({}):", tasks.len());
        for t in tasks.iter().take(10) {
            println!("    - {}", t);
        }
        if tasks.len() > 10 {
            println!("    ... and {} more", tasks.len() - 10);
        }

        if success_known > 0 {
            let rate = success_count as f64 / success_known as f64 * 100.0;
            println!();
            println!(
                "  Success:   {}/{} ({:.1}%)",
                success_count, success_known, rate
            );
        }
    }

    // ── Per-episode table ───────────────────────────────────
    if show_episodes {
        println!();
        println!(
            "  {:<6} {:<10} {:<8} {:<10} {:<10} TASK",
            "ID", "EMBODIMENT", "FRAMES", "ACT_DIM", "STATE_DIM"
        );
        println!("  {}", "-".repeat(70));

        for i in 0..reader.num_episodes() {
            let entry = reader.index().get(i).unwrap();
            let task = reader
                .read_meta(i)
                .map(|m| m.language_instruction)
                .unwrap_or_else(|_| "???".to_string());

            // Truncate task to 30 chars
            let task_display = if task.len() > 30 {
                format!("{}...", &task[..27])
            } else {
                task
            };

            let embodiment = reader
                .read_meta(i)
                .map(|m| m.embodiment)
                .unwrap_or_else(|_| "???".to_string());

            println!(
                "  {:<6} {:<10} {:<8} {:<10} {:<10} {}",
                entry.episode_id.0,
                embodiment,
                entry.num_frames,
                entry.action_dim,
                entry.state_dim,
                task_display,
            );
        }
    }

    Ok(())
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Format a unix timestamp as a readable date string.
/// We do this without chrono — just a basic UTC display.
fn format_timestamp(secs: u64) -> String {
    if secs == 0 {
        return "unknown".to_string();
    }

    // Basic: just show as unix timestamp with a note.
    // We could pull in chrono later, but keeping deps minimal.
    //
    // For a rough human-readable version, compute year/month/day
    // from the unix timestamp manually (good enough for display).
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;

    // Days since 1970-01-01. Walk through years.
    let mut remaining_days = days;
    let mut year: u64 = 1970;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    // Walk through months
    let days_in_months: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month: u64 = 1;
    for &dm in &days_in_months {
        if remaining_days < dm {
            break;
        }
        remaining_days -= dm;
        month += 1;
    }
    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02} UTC",
        year, month, day, hours, minutes
    )
}

fn is_leap(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}
