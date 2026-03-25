use clap::{Parser, Subcommand};
use std::process;

mod cmd_info;
mod cmd_create_test;

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
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}
