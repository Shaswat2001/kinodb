use kinodb_core::KdbReader;
use std::collections::{BTreeMap, BTreeSet};

pub fn run(kdb_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let reader = KdbReader::open(kdb_path)?;
    let header = reader.header();

    println!("Schema: {}", kdb_path);
    println!();

    // ── File-level info ─────────────────────────────────────
    println!("Format");
    println!(
        "  version:    {}.{}",
        header.version_major, header.version_minor
    );
    println!("  episodes:   {}", header.num_episodes);
    println!("  frames:     {}", header.num_frames);

    let file_size = std::fs::metadata(kdb_path)?.len();
    println!(
        "  file_size:  {} ({} bytes)",
        format_bytes(file_size),
        file_size
    );
    println!("  index_at:   byte {}", header.index_offset);
    println!("  index_size: {} bytes", header.index_length);
    println!();

    if reader.num_episodes() == 0 {
        println!("  (empty database)");
        return Ok(());
    }

    // ── Scan episodes to discover schema ────────────────────
    let mut embodiments: BTreeSet<String> = BTreeSet::new();
    let mut tasks: BTreeSet<String> = BTreeSet::new();
    let mut action_dims: BTreeSet<u16> = BTreeSet::new();
    let mut state_dims: BTreeSet<usize> = BTreeSet::new();
    let mut fps_values: BTreeSet<String> = BTreeSet::new();
    let mut frame_counts: Vec<u32> = Vec::new();
    let mut cameras: BTreeMap<String, (u32, u32, u8)> = BTreeMap::new(); // name -> (w, h, c)
    let mut has_images = false;
    let mut has_rewards = false;
    let mut has_success = false;
    let mut success_count = 0u64;
    let mut total_reward_sum = 0.0f64;
    let mut total_reward_count = 0u64;

    // Read a sample episode for detailed structure
    let sample_ep = reader.read_episode(0)?;

    for i in 0..reader.num_episodes() {
        let meta = reader.read_meta(i)?;
        embodiments.insert(meta.embodiment.clone());
        tasks.insert(meta.language_instruction.clone());
        action_dims.insert(meta.action_dim);
        fps_values.insert(format!("{:.1}", meta.fps));
        frame_counts.push(meta.num_frames);

        if meta.success.is_some() {
            has_success = true;
            if meta.success == Some(true) {
                success_count += 1;
            }
        }
        if let Some(r) = meta.total_reward {
            has_rewards = true;
            total_reward_sum += r as f64;
            total_reward_count += 1;
        }
    }

    // Get state dim and camera info from sample episode
    if !sample_ep.frames.is_empty() {
        let f0 = &sample_ep.frames[0];
        state_dims.insert(f0.state.len());

        for img in &f0.images {
            has_images = true;
            cameras.insert(img.camera.clone(), (img.width, img.height, img.channels));
        }
    }

    // Scan a few more episodes for state_dim and camera consistency
    let scan_count = std::cmp::min(10, reader.num_episodes());
    for i in 1..scan_count {
        if let Ok(ep) = reader.read_episode(i) {
            if let Some(f) = ep.frames.first() {
                state_dims.insert(f.state.len());
                for img in &f.images {
                    cameras.entry(img.camera.clone()).or_insert((
                        img.width,
                        img.height,
                        img.channels,
                    ));
                }
            }
        }
    }

    // ── Print schema ────────────────────────────────────────
    println!("Episode Fields");
    println!("  embodiment:           {:?}", set_summary(&embodiments));
    println!("  language_instruction: {} unique values", tasks.len());
    println!("  fps:                  {:?}", set_summary(&fps_values));
    println!(
        "  success:              {}",
        if has_success { "present" } else { "absent" }
    );
    println!(
        "  total_reward:         {}",
        if has_rewards { "present" } else { "absent" }
    );
    println!();

    // Frame length stats
    let min_frames = frame_counts.iter().min().copied().unwrap_or(0);
    let max_frames = frame_counts.iter().max().copied().unwrap_or(0);
    let avg_frames = if !frame_counts.is_empty() {
        frame_counts.iter().map(|&x| x as f64).sum::<f64>() / frame_counts.len() as f64
    } else {
        0.0
    };

    println!("Frame Layout");
    println!(
        "  num_frames:  min={}, max={}, avg={:.1}",
        min_frames, max_frames, avg_frames
    );
    println!("  action:      {} x f32", format_set_u16(&action_dims));
    println!("  state:       {} x f32", format_set_usize(&state_dims));
    println!("  reward:      f32 per frame");
    println!("  is_terminal: bool per frame");
    println!();

    if has_images {
        println!("Cameras ({})", cameras.len());
        for (name, (w, h, c)) in &cameras {
            let bytes_per_frame = (*w as usize) * (*h as usize) * (*c as usize);
            println!(
                "  {:<30} {}x{}x{} uint8 ({}/frame)",
                name,
                h,
                w,
                c,
                format_bytes(bytes_per_frame as u64),
            );
        }
        println!();
    } else {
        println!("Cameras: none");
        println!();
    }

    // ── Statistics ───────────────────────────────────────────
    if has_success || has_rewards {
        println!("Statistics");
        if has_success {
            let total = reader.num_episodes();
            let rate = if total > 0 {
                success_count as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            println!("  success_rate: {}/{} ({:.1}%)", success_count, total, rate);
        }
        if has_rewards && total_reward_count > 0 {
            let avg_reward = total_reward_sum / total_reward_count as f64;
            println!("  avg_reward:   {:.3}", avg_reward);
        }
        println!();
    }

    // ── Per-episode byte budget ─────────────────────────────
    let index = reader.index();
    if reader.num_episodes() > 0 {
        let entry = index.get(0).unwrap();
        println!("Byte Budget (episode 0)");
        println!("  meta:    {} bytes", entry.meta_length);
        println!(
            "  actions: {} ({}/frame)",
            format_bytes(entry.actions_length),
            format_bytes(entry.actions_length / entry.num_frames as u64),
        );
        println!(
            "  images:  {} ({}/frame)",
            format_bytes(entry.images_length),
            if entry.num_frames > 0 {
                format_bytes(entry.images_length / entry.num_frames as u64)
            } else {
                "0 B".to_string()
            },
        );
        let total_ep = entry.meta_length + entry.actions_length + entry.images_length;
        println!("  total:   {}", format_bytes(total_ep));
        println!();
    }

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

fn set_summary(s: &BTreeSet<String>) -> String {
    let items: Vec<&str> = s.iter().map(|s| s.as_str()).collect();
    if items.len() <= 4 {
        format!("{}", items.join(", "))
    } else {
        format!("{}, ... ({} total)", items[..3].join(", "), items.len())
    }
}

fn format_set_u16(s: &BTreeSet<u16>) -> String {
    let items: Vec<String> = s.iter().map(|x| x.to_string()).collect();
    items.join(" | ")
}

fn format_set_usize(s: &BTreeSet<usize>) -> String {
    let items: Vec<String> = s.iter().map(|x| x.to_string()).collect();
    items.join(" | ")
}
