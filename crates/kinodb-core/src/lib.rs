//! # kinodb-core
//!
//! Core storage engine for **kinodb** — a high-performance trajectory
//! database for robot learning.
//!
//! This crate provides:
//! - The `.kdb` file format (read + write)
//! - Episode, frame, and action storage
//! - Memory-mapped access for zero-copy reads
//!
//! ## Status
//! 🚧 Early development — API will change.

mod header;
mod index;
mod mixture;
mod reader;
mod types;
mod writer;

pub use header::*;
pub use index::*;
pub use mixture::*;
pub use reader::*;
pub use types::*;
pub use writer::*;
