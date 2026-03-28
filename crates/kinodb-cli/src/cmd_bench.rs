use kinodb_core::{kql, Episode, EpisodeId, EpisodeMeta, Frame, ImageObs, KdbReader, KdbWriter};
use std::time::Instant;

pub fn run(
    num_episodes: u32,
    frames_per_ep: u32,
    with_images: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = "/tmp/kinodb_bench.kdb";

    println!("kinodb benchmark");
    println!("================");
    println!("  Episodes:     {}", num_episodes);
    println!("  Frames/ep:    {}", frames_per_ep);
    println!(
        "  Images:       {}",
        if with_images {
            "64x64 RGB (1 camera)"
        } else {
            "none"
        }
    );
    println!();

    // ── 1. Write benchmark ──────────────────────────────────
    let episodes: Vec<Episode> = (0..num_episodes)
        .map(|i| make_bench_episode(i, frames_per_ep, with_images))
        .collect();

    let t0 = Instant::now();
    let mut writer = KdbWriter::create(path)?;
    for ep in &episodes {
        writer.write_episode(ep)?;
    }
    writer.finish()?;
    let write_dur = t0.elapsed();

    let file_size = std::fs::metadata(path)?.len();
    let total_frames = num_episodes as u64 * frames_per_ep as u64;

    println!("WRITE");
    println!("  Time:         {:.1} ms", write_dur.as_secs_f64() * 1000.0);
    println!(
        "  Throughput:   {:.0} episodes/sec",
        num_episodes as f64 / write_dur.as_secs_f64()
    );
    println!(
        "  Throughput:   {:.0} frames/sec",
        total_frames as f64 / write_dur.as_secs_f64()
    );
    println!("  File size:    {}", format_bytes(file_size));
    if total_frames > 0 {
        println!("  Per frame:    {} bytes", file_size / total_frames);
    }
    println!();

    // ── 2. Open benchmark ───────────────────────────────────
    let t0 = Instant::now();
    let reader = KdbReader::open(path)?;
    let open_dur = t0.elapsed();

    println!("OPEN");
    println!("  Time:         {:.2} ms", open_dur.as_secs_f64() * 1000.0);
    println!();

    // ── 3. Sequential read (all episodes) ───────────────────
    let t0 = Instant::now();
    let mut frames_read = 0u64;
    for i in 0..reader.num_episodes() {
        let ep = reader.read_episode(i)?;
        frames_read += ep.frames.len() as u64;
    }
    let seq_dur = t0.elapsed();

    println!("SEQUENTIAL READ (all {} episodes)", reader.num_episodes());
    println!("  Time:         {:.1} ms", seq_dur.as_secs_f64() * 1000.0);
    println!(
        "  Throughput:   {:.0} episodes/sec",
        reader.num_episodes() as f64 / seq_dur.as_secs_f64()
    );
    println!(
        "  Throughput:   {:.0} frames/sec",
        frames_read as f64 / seq_dur.as_secs_f64()
    );
    println!();

    // ── 4. Random read (1000 random episodes) ───────────────
    let n_random = std::cmp::min(1000, reader.num_episodes());
    let random_indices: Vec<usize> = (0..n_random)
        .map(|i| (i * 7 + 13) % reader.num_episodes()) // deterministic pseudo-random
        .collect();

    let t0 = Instant::now();
    let mut rand_frames = 0u64;
    for &idx in &random_indices {
        let ep = reader.read_episode(idx)?;
        rand_frames += ep.frames.len() as u64;
    }
    let rand_dur = t0.elapsed();

    println!("RANDOM READ ({} episodes)", n_random);
    println!("  Time:         {:.1} ms", rand_dur.as_secs_f64() * 1000.0);
    println!(
        "  Throughput:   {:.0} episodes/sec",
        n_random as f64 / rand_dur.as_secs_f64()
    );
    println!(
        "  Throughput:   {:.0} frames/sec",
        rand_frames as f64 / rand_dur.as_secs_f64()
    );
    println!();

    // ── 5. Metadata-only read ───────────────────────────────
    let t0 = Instant::now();
    for i in 0..reader.num_episodes() {
        let _meta = reader.read_meta(i)?;
    }
    let meta_dur = t0.elapsed();

    println!("METADATA READ (all {} episodes)", reader.num_episodes());
    println!("  Time:         {:.2} ms", meta_dur.as_secs_f64() * 1000.0);
    println!(
        "  Throughput:   {:.0} episodes/sec",
        reader.num_episodes() as f64 / meta_dur.as_secs_f64()
    );
    println!();

    // ── 6. KQL query benchmark ──────────────────────────────
    let queries = [
        "embodiment = 'franka'",
        "success = true",
        "num_frames > 25",
        "embodiment = 'franka' AND success = true AND num_frames > 25",
        "task CONTAINS 'pick'",
    ];

    println!(
        "KQL QUERY ({} episodes scanned per query)",
        reader.num_episodes()
    );
    for query_str in &queries {
        let t0 = Instant::now();
        let query = kql::parse(query_str).unwrap();
        let mut hits = 0;
        for i in 0..reader.num_episodes() {
            let meta = reader.read_meta(i)?;
            if kql::evaluate(&query, &meta) {
                hits += 1;
            }
        }
        let q_dur = t0.elapsed();

        println!(
            "  {:<55} {:>5} hits  {:.2} ms",
            format!("\"{}\"", query_str),
            hits,
            q_dur.as_secs_f64() * 1000.0,
        );
    }
    println!();

    // ── Cleanup ─────────────────────────────────────────────
    std::fs::remove_file(path).ok();

    println!("Done.");
    Ok(())
}

fn make_bench_episode(idx: u32, num_frames: u32, with_images: bool) -> Episode {
    let embodiments = ["franka", "widowx", "aloha", "ur5"];
    let tasks = [
        "pick up the red block",
        "open the drawer",
        "place cup on plate",
        "push the button",
    ];

    let embodiment = embodiments[idx as usize % embodiments.len()];
    let task = tasks[idx as usize % tasks.len()];
    let success = idx % 3 != 0;

    let meta = EpisodeMeta {
        id: EpisodeId(idx as u64),
        embodiment: embodiment.to_string(),
        language_instruction: task.to_string(),
        num_frames,
        fps: 10.0,
        action_dim: 7,
        success: Some(success),
        total_reward: if success { Some(1.0) } else { Some(0.0) },
    };

    let frames = (0..num_frames)
        .map(|t| {
            let images = if with_images {
                vec![ImageObs {
                    camera: "front".to_string(),
                    width: 64,
                    height: 64,
                    channels: 3,
                    data: vec![128u8; 64 * 64 * 3],
                }]
            } else {
                vec![]
            };

            Frame {
                timestep: t,
                images,
                state: vec![0.1; 6],
                action: vec![0.01; 7],
                reward: Some(if t == num_frames - 1 && success {
                    1.0
                } else {
                    0.0
                }),
                is_terminal: t == num_frames - 1,
            }
        })
        .collect();

    Episode { meta, frames }
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
