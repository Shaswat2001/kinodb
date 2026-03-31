use kinodb_core::{KdbReader, KdbWriter, kql};

pub fn run(
    inputs: &[String],
    output: &str,
    filter: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    if inputs.is_empty() {
        eprintln!("No input files provided.");
        std::process::exit(1);
    }

    println!("Merging {} files into {}", inputs.len(), output);

    // Parse filter if provided
    let query = match filter {
        Some(q) => {
            println!("  Filter: {}", q);
            Some(kql::parse(q)?)
        }
        None => None,
    };
    println!();

    let mut writer = KdbWriter::create(output)?;
    let mut total_episodes = 0u64;
    let mut total_frames = 0u64;
    let mut skipped = 0u64;

    for input_path in inputs {
        let reader = KdbReader::open(input_path)?;
        let n = reader.num_episodes();
        let mut file_written = 0u64;

        for i in 0..n {
            // If filter is set, check metadata first (cheap)
            if let Some(ref q) = query {
                let meta = reader.read_meta(i)?;
                if !kql::evaluate(q, &meta) {
                    skipped += 1;
                    continue;
                }
            }

            let episode = reader.read_episode(i)?;
            writer.write_episode(&episode)?;
            total_frames += episode.frames.len() as u64;
            total_episodes += 1;
            file_written += 1;
        }

        println!(
            "  {} — {}/{} episodes written",
            input_path, file_written, n
        );
    }

    writer.finish()?;

    let file_size = std::fs::metadata(output)?.len();

    println!();
    println!("Done!");
    println!("  Episodes: {}", total_episodes);
    println!("  Frames:   {}", total_frames);
    if skipped > 0 {
        println!("  Skipped:  {} (filtered out)", skipped);
    }
    println!("  Size:     {}", format_bytes(file_size));
    println!();
    println!("Try:");
    println!("  kino info {}", output);

    Ok(())
}

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
