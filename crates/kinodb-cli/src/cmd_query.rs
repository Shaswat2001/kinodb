use kinodb_core::kql;
use kinodb_core::KdbReader;

pub fn run(
    kdb_path: &str,
    query_str: &str,
    limit: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = KdbReader::open(kdb_path)?;

    // Parse and validate the query first
    let query = kql::parse(query_str)?;

    println!("Query: {}", query_str);
    println!("Scanning {} episodes...", reader.num_episodes());
    println!();

    let mut matches = Vec::new();
    for i in 0..reader.num_episodes() {
        let meta = reader.read_meta(i)?;
        if kql::evaluate(&query, &meta) {
            matches.push((i, meta));
        }
        if let Some(lim) = limit {
            if matches.len() >= lim {
                break;
            }
        }
    }

    println!("Found {} matching episodes.", matches.len());

    if !matches.is_empty() {
        println!();
        println!(
            "  {:<6} {:<12} {:<8} {:<8} {:<8} {}",
            "ID", "EMBODIMENT", "FRAMES", "ACT", "SUCCESS", "TASK"
        );
        println!("  {}", "-".repeat(70));

        for (pos, meta) in &matches {
            let task = if meta.language_instruction.len() > 28 {
                format!("{}...", &meta.language_instruction[..25])
            } else {
                meta.language_instruction.clone()
            };

            let success = match meta.success {
                Some(true) => "yes",
                Some(false) => "no",
                None => "-",
            };

            println!(
                "  {:<6} {:<12} {:<8} {:<8} {:<8} {}",
                pos, meta.embodiment, meta.num_frames, meta.action_dim, success, task
            );
        }
    }

    Ok(())
}
