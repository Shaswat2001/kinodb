use kinodb_core::Mixture;

pub fn run(
    sources: &[(String, f64)],
    seed: u64,
    sample_n: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    if sources.is_empty() {
        eprintln!("No sources provided. Usage:");
        eprintln!("  kino mix --source bridge.kdb:0.4 --source aloha.kdb:0.6");
        std::process::exit(1);
    }

    println!(
        "Building mixture ({} sources, seed={}):",
        sources.len(),
        seed
    );

    let mut builder = Mixture::builder().seed(seed);
    for (path, weight) in sources {
        builder = builder.add(path, *weight);
        println!("  {}: weight={:.2}", path, weight);
    }

    let mut mix = builder.build()?;
    println!();

    // Print summary
    let info = mix.source_info();
    let total_ep = mix.total_episodes();
    let total_fr = mix.total_frames();

    println!("Mixture summary:");
    println!("  Total episodes: {}", total_ep);
    println!("  Total frames:   {}", total_fr);
    println!();
    println!(
        "  {:<30} {:>10} {:>10} {:>12}",
        "SOURCE", "EPISODES", "WEIGHT", "EFF. SHARE"
    );
    println!("  {}", "-".repeat(65));

    for (path, num_ep, weight) in &info {
        println!(
            "  {:<30} {:>10} {:>9.1}% {:>11.1}%",
            truncate_path(path, 30),
            num_ep,
            weight * 100.0,
            weight * 100.0,
        );
    }

    // Optional: sample and show distribution
    if let Some(n) = sample_n {
        println!();
        println!("Sampling {} episodes...", n);

        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for _ in 0..n {
            let meta = mix.sample_meta()?;
            *counts.entry(meta.embodiment.clone()).or_insert(0) += 1;
        }

        println!();
        println!("  Empirical distribution:");
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (embodiment, count) in &sorted {
            let pct = (*count as f64 / n as f64) * 100.0;
            println!("    {}: {} ({:.1}%)", embodiment, count, pct);
        }
    }

    Ok(())
}

fn truncate_path(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("...{}", &s[s.len() - max + 3..])
    }
}
