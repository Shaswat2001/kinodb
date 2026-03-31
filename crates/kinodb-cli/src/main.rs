use clap::{Parser, Subcommand};
use std::process;

mod cmd_bench;
mod cmd_create_test;
mod cmd_export;
mod cmd_info;
mod cmd_ingest;
mod cmd_merge;
mod cmd_mix;
mod cmd_query;
mod cmd_schema;
mod cmd_validate;

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

        /// JPEG compress images (quality 1-100, recommended 85).
        #[arg(long)]
        compress: Option<u8>,
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

        /// JPEG compress images (quality 1-100, recommended 85).
        #[arg(long)]
        compress: Option<u8>,
    },

    /// Export a .kdb file to standard formats (numpy binary + JSON).
    Export {
        /// Path to the .kdb file.
        kdb_path: String,

        /// Output directory.
        #[arg(short, long, default_value = "export")]
        output: String,

        /// Export format: numpy, json.
        #[arg(short = 'F', long, default_value = "numpy")]
        format: String,
    },

    /// Create and inspect weighted dataset mixtures.
    Mix {
        /// Sources in "path:weight" format. Can be repeated.
        /// Example: --source bridge.kdb:0.4 --source aloha.kdb:0.6
        #[arg(short, long, required = true, value_parser = parse_source)]
        source: Vec<(String, f64)>,

        /// Random seed for sampling.
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Sample N episodes and show the empirical distribution.
        #[arg(long)]
        sample: Option<usize>,
    },

    /// Filter episodes using KQL (Kino Query Language).
    Query {
        /// Path to the .kdb file.
        kdb_path: String,

        /// KQL query string. Example: "embodiment = 'franka' AND success = true"
        query: String,

        /// Maximum number of results to show.
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Run performance benchmarks (write, read, query).
    Bench {
        /// Number of episodes to generate for the benchmark.
        #[arg(short = 'n', long, default_value = "500")]
        num_episodes: u32,

        /// Frames per episode.
        #[arg(short, long, default_value = "50")]
        frames: u32,

        /// Include 64x64 RGB images in the benchmark.
        #[arg(long)]
        images: bool,
    },

    /// Print the schema and structure of a .kdb file.
    Schema {
        /// Path to the .kdb file.
        path: String,
    },

    /// Validate a .kdb file for corruption and consistency issues.
    Validate {
        /// Path to the .kdb file.
        path: String,

        /// Show all warnings and errors (not just first 10).
        #[arg(short, long)]
        verbose: bool,
    },

    /// Merge multiple .kdb files into one. Optionally filter with KQL.
    Merge {
        /// Input .kdb files to merge.
        #[arg(required = true)]
        inputs: Vec<String>,

        /// Output .kdb file path.
        #[arg(short, long, default_value = "merged.kdb")]
        output: String,

        /// Optional KQL filter — only episodes matching the query are included.
        #[arg(short = 'F', long)]
        filter: Option<String>,
    },
}

/// Parse "path:weight" string into (String, f64).
fn parse_source(s: &str) -> Result<(String, f64), String> {
    if let Some((path, weight_str)) = s.rsplit_once(':') {
        let weight: f64 = weight_str
            .parse()
            .map_err(|_| format!("invalid weight '{}' in '{}'", weight_str, s))?;
        Ok((path.to_string(), weight))
    } else {
        // No weight specified — default to 1.0
        Ok((s.to_string(), 1.0))
    }
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
            compress,
        } => cmd_create_test::run(&path, num_episodes, frames, images, compress),
        Commands::Ingest {
            src,
            output,
            format,
            embodiment,
            task,
            fps,
            max_episodes,
            compress,
        } => cmd_ingest::run(
            &src,
            &output,
            &format,
            &embodiment,
            task.as_deref(),
            fps,
            max_episodes,
            compress,
        ),
        Commands::Export {
            kdb_path,
            output,
            format,
        } => cmd_export::run(&kdb_path, &output, &format),
        Commands::Mix {
            source,
            seed,
            sample,
        } => cmd_mix::run(&source, seed, sample),
        Commands::Query {
            kdb_path,
            query,
            limit,
        } => cmd_query::run(&kdb_path, &query, limit),
        Commands::Bench {
            num_episodes,
            frames,
            images,
        } => cmd_bench::run(num_episodes, frames, images),
        Commands::Schema { path } => cmd_schema::run(&path),
        Commands::Validate { path, verbose } => cmd_validate::run(&path, verbose),
        Commands::Merge {
            inputs,
            output,
            filter,
        } => cmd_merge::run(&inputs, &output, filter.as_deref()),
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}
