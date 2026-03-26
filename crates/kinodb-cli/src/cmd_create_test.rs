use kinodb_core::{Episode, EpisodeId, EpisodeMeta, Frame, ImageObs, KdbWriter};

pub fn run(
    path: &str,
    num_episodes: u32,
    frames_per_episode: u32,
    with_images: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating test database: {}", path);
    println!(
        "  {} episodes x {} frames{}",
        num_episodes,
        frames_per_episode,
        if with_images { " (with 2 cameras)" } else { "" }
    );

    let mut writer = KdbWriter::create(path)?;

    let embodiments = ["widowx", "franka", "aloha", "ur5"];
    let tasks = [
        "pick up the red block",
        "open the drawer",
        "place the cup on the plate",
        "push the green button",
        "stack the blue cube on the yellow cube",
        "close the laptop lid",
    ];

    for ep_idx in 0..num_episodes {
        let embodiment = embodiments[ep_idx as usize % embodiments.len()];
        let task = tasks[ep_idx as usize % tasks.len()];
        let success = ep_idx % 3 != 0; // ~66% success rate

        let meta = EpisodeMeta {
            id: EpisodeId(0), // writer assigns its own
            embodiment: embodiment.to_string(),
            language_instruction: task.to_string(),
            num_frames: frames_per_episode,
            fps: 10.0,
            action_dim: 7,
            success: Some(success),
            total_reward: if success { Some(1.0) } else { Some(0.0) },
        };

        let frames: Vec<Frame> = (0..frames_per_episode)
            .map(|t| {
                let progress = t as f32 / frames_per_episode as f32;

                let images = if with_images {
                    vec![
                        make_fake_image("front", 64, 64, progress),
                        make_fake_image("wrist", 64, 64, progress),
                    ]
                } else {
                    vec![]
                };

                Frame {
                    timestep: t,
                    images,
                    // 6-DoF state: simulate a smooth trajectory
                    state: vec![
                        progress * 0.3,       // x
                        progress * 0.1,       // y
                        0.5 + progress * 0.2, // z
                        0.0,                  // roll
                        0.0,                  // pitch
                        progress * 0.1,       // yaw
                    ],
                    // 7-DoF action: small deltas
                    action: vec![
                        0.01,                                               // dx
                        0.005,                                              // dy
                        0.01,                                               // dz
                        0.0,                                                // droll
                        0.0,                                                // dpitch
                        0.005,                                              // dyaw
                        if t < frames_per_episode - 2 { 0.0 } else { 1.0 }, // gripper
                    ],
                    reward: Some(if t == frames_per_episode - 1 && success {
                        1.0
                    } else {
                        0.0
                    }),
                    is_terminal: t == frames_per_episode - 1,
                }
            })
            .collect();

        let episode = Episode { meta, frames };
        writer.write_episode(&episode)?;

        // Progress indicator for large datasets
        if num_episodes >= 100 && (ep_idx + 1) % 100 == 0 {
            println!("  wrote {}/{} episodes", ep_idx + 1, num_episodes);
        }
    }

    writer.finish()?;

    // Print result
    let file_size = std::fs::metadata(path)?.len();
    println!();
    println!("Done! Wrote {}", format_bytes(file_size));
    println!();
    println!("Try it:");
    println!("  kino info {}", path);
    println!("  kino info --episodes {}", path);

    Ok(())
}

/// Create a fake image with a simple gradient pattern.
/// Not meant to look realistic — just deterministic test data
/// that's easy to verify survived a roundtrip.
fn make_fake_image(camera: &str, width: u32, height: u32, progress: f32) -> ImageObs {
    let channels: u8 = 3;
    let size = (width * height * channels as u32) as usize;
    let mut data = Vec::with_capacity(size);

    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = (progress * 255.0) as u8;
            data.push(r);
            data.push(g);
            data.push(b);
        }
    }

    ImageObs {
        camera: camera.to_string(),
        width,
        height,
        channels,
        data,
    }
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
