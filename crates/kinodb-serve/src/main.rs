//! `kino-serve` — gRPC batch server for kinodb.
//!
//! Serves trajectory data from one or more .kdb files over gRPC.
//! Training scripts connect as clients to request episodes and batches.
//!
//! ## Usage
//!
//! ```bash
//! # Single file
//! kino-serve data.kdb --port 50051
//!
//! # Weighted mixture
//! kino-serve --source bridge.kdb:0.4 --source aloha.kdb:0.6 --port 50051
//! ```

mod server;

use clap::Parser;
use tonic::transport::Server;

use server::pb::kino_service_server::KinoServiceServer;
use server::KinoServer;

#[derive(Parser)]
#[command(name = "kino-serve", about = "gRPC batch server for kinodb")]
struct Cli {
    /// Single .kdb file to serve.
    #[arg(value_name = "FILE")]
    file: Option<String>,

    /// Sources in "path:weight" format (for mixture serving).
    /// Example: --source bridge.kdb:0.4 --source aloha.kdb:0.6
    #[arg(short, long, value_parser = parse_source)]
    source: Vec<(String, f64)>,

    /// Port to listen on.
    #[arg(short, long, default_value = "50051")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    bind: String,

    /// Random seed for mixture sampling.
    #[arg(long, default_value = "42")]
    seed: u64,
}

fn parse_source(s: &str) -> Result<(String, f64), String> {
    if let Some((path, weight_str)) = s.rsplit_once(':') {
        let weight: f64 = weight_str
            .parse()
            .map_err(|_| format!("invalid weight '{}' in '{}'", weight_str, s))?;
        Ok((path.to_string(), weight))
    } else {
        Ok((s.to_string(), 1.0))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let server = if let Some(ref file) = cli.file {
        // Single file mode
        println!("kinodb gRPC server");
        println!("  Mode:    single file");
        println!("  File:    {}", file);
        KinoServer::from_single(file)?
    } else if !cli.source.is_empty() {
        // Mixture mode
        println!("kinodb gRPC server");
        println!("  Mode:    weighted mixture");
        for (path, weight) in &cli.source {
            println!("  Source:  {} (weight: {:.2})", path, weight);
        }
        KinoServer::from_mixture(&cli.source, cli.seed)?
    } else {
        eprintln!("Error: provide a .kdb file or --source arguments");
        eprintln!();
        eprintln!("Usage:");
        eprintln!("  kino-serve data.kdb --port 50051");
        eprintln!("  kino-serve --source bridge.kdb:0.4 --source aloha.kdb:0.6");
        std::process::exit(1);
    };

    let addr = format!("{}:{}", cli.bind, cli.port).parse()?;
    println!("  Listen:  {}", addr);
    println!();

    Server::builder()
        .add_service(KinoServiceServer::new(server))
        .serve(addr)
        .await?;

    Ok(())
}
