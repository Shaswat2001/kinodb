use kinodb_ingest::hdf5::{self, Hdf5IngestConfig};
use kinodb_ingest::lerobot::{self, LeRobotIngestConfig};
use kinodb_ingest::rlds::{self, RldsIngestConfig};

pub fn run(
    src: &str,
    output: &str,
    format: &str,
    embodiment: &str,
    task: Option<&str>,
    fps: f32,
    max_episodes: Option<usize>,
    compress: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        "hdf5" => run_hdf5(src, output, embodiment, task, fps, max_episodes, compress),
        "lerobot" => run_lerobot(src, output, embodiment, task, max_episodes, compress),
        "rlds" | "tfrecord" => run_rlds(src, output, embodiment, task, fps, max_episodes),
        other => {
            eprintln!("Unsupported format: '{}'. Supported: hdf5, lerobot, rlds", other);
            std::process::exit(1);
        }
    }
}

fn run_hdf5(
    src: &str,
    output: &str,
    embodiment: &str,
    task: Option<&str>,
    fps: f32,
    max_episodes: Option<usize>,
    compress: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Ingesting HDF5: {}", src);
    println!("  Output:     {}", output);
    println!("  Embodiment: {}", embodiment);
    if let Some(t) = task {
        println!("  Task:       {}", t);
    }
    println!("  FPS:        {}", fps);
    if let Some(max) = max_episodes {
        println!("  Max episodes: {}", max);
    }
    if let Some(q) = compress {
        println!("  Compress:   JPEG quality {}", q);
    }
    println!();

    let config = Hdf5IngestConfig {
        embodiment: embodiment.to_string(),
        task: task.map(|s| s.to_string()),
        fps,
        max_episodes,
        compress,
    };

    let result = hdf5::ingest_hdf5(src, output, &config)?;

    println!(
        "Done! Wrote {} episodes ({} frames) to {}",
        result.num_episodes, result.total_frames, result.output_path
    );
    println!();
    println!("Try:");
    println!("  kino info {}", output);

    Ok(())
}

fn run_lerobot(
    src: &str,
    output: &str,
    embodiment: &str,
    task: Option<&str>,
    max_episodes: Option<usize>,
    compress: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Ingesting LeRobot dataset: {}", src);
    println!("  Output: {}", output);
    if embodiment != "unknown" {
        println!("  Embodiment: {}", embodiment);
    }
    if let Some(t) = task {
        println!("  Task: {}", t);
    }
    if let Some(max) = max_episodes {
        println!("  Max episodes: {}", max);
    }
    if let Some(q) = compress {
        println!("  Compress: JPEG quality {}", q);
    }
    println!();

    let config = LeRobotIngestConfig {
        embodiment: if embodiment == "unknown" { None } else { Some(embodiment.to_string()) },
        task: task.map(|s| s.to_string()),
        max_episodes,
        compress,
    };

    let result = lerobot::ingest_lerobot(src, output, &config)?;

    println!(
        "Done! Wrote {} episodes ({} frames) to {}",
        result.num_episodes, result.total_frames, result.output_path
    );
    println!();
    println!("Try:");
    println!("  kino info {}", output);
    println!("  kino schema {}", output);

    Ok(())
}

fn run_rlds(
    src: &str,
    output: &str,
    embodiment: &str,
    task: Option<&str>,
    fps: f32,
    max_episodes: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Ingesting RLDS (TFRecord): {}", src);
    println!("  Output: {}", output);
    println!("  Embodiment: {}", embodiment);
    if let Some(t) = task {
        println!("  Task: {}", t);
    }
    println!("  FPS: {}", fps);
    if let Some(max) = max_episodes {
        println!("  Max episodes: {}", max);
    }
    println!();

    let config = RldsIngestConfig {
        embodiment: embodiment.to_string(),
        task: task.map(|s| s.to_string()),
        fps,
        max_episodes,
    };

    let result = rlds::ingest_rlds(src, output, &config)?;

    println!(
        "Done! Wrote {} episodes ({} frames) to {}",
        result.num_episodes, result.total_frames, result.output_path
    );
    println!();
    println!("Try:");
    println!("  kino info {}", output);

    Ok(())
}