use clap::{Parser, Subcommand};
use std::process;

mod cmd_info;
mod cmd_create_test;
mod cmd_ingest;

/// kinodb — a high-performance trajectory database for robot learning.
#[derive(Parser)]
#[command(name = "kino", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show summary information about a .kdb file.
    Info {
        /// Path to the .kdb file.
        path: String,

        /// Show per-episode details (not just the summary).
        #[arg(short, long)]
        episodes: bool,
    },

    /// Generate a sample .kdb file with fake robot data (for testing).
    CreateTest {
        /// Output path for the .kdb file.
        #[arg(default_value = "test.kdb")]
        path: String,

        /// Number of episodes to generate.
        #[arg(short = 'n', long, default_value = "10")]
        num_episodes: u32,

        /// Number of frames per episode.
        #[arg(short, long, default_value = "50")]
        frames: u32,

        /// Include fake 64x64 camera images (makes file much larger).
        #[arg(long)]
        images: bool,
    },

    /// Ingest trajectory data from external formats into a .kdb file.
    Ingest {
        /// Path to the source file (e.g. an HDF5 file).
        src: String,

        /// Output .kdb file path.
        #[arg(short, long, default_value = "output.kdb")]
        output: String,

        /// Source format: hdf5, lerobot, rlds.
        #[arg(short = 'F', long, default_value = "hdf5")]
        format: String,

        /// Robot embodiment name (e.g. "franka", "widowx", "aloha").
        #[arg(short, long, default_value = "unknown")]
        embodiment: String,

        /// Task description (auto-detected from filename if omitted).
        #[arg(short, long)]
        task: Option<String>,

        /// Control frequency in Hz.
        #[arg(long, default_value = "10.0")]
        fps: f32,

        /// Only ingest the first N episodes.
        #[arg(long)]
        max_episodes: Option<usize>,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Info { path, episodes } => cmd_info::run(&path, episodes),
        Commands::CreateTest {
            path,
            num_episodes,
            frames,
            images,
        } => cmd_create_test::run(&path, num_episodes, frames, images),
        Commands::Ingest {
            src,
            output,
            format,
            embodiment,
            task,
            fps,
            max_episodes,
        } => cmd_ingest::run(
            &src,
            &output,
            &format,
            &embodiment,
            task.as_deref(),
            fps,
            max_episodes,
        ),
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}
