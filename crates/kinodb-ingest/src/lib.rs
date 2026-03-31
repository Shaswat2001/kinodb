//! # kinodb-ingest
//!
//! Format ingesters for kinodb. Reads robot trajectory data from
//! various formats and writes `.kdb` files.
//!
//! ## Supported formats
//!
//! - **HDF5** (robomimic / LIBERO / DROID style) — [`hdf5::ingest_hdf5`]
//! - **LeRobot** v2/v3 (Parquet + MP4) — [`lerobot::ingest_lerobot`]
//! - RLDS — planned

pub mod hdf5;
pub mod lerobot;
pub mod rlds;
