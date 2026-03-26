//! # kinodb-ingest
//!
//! Format ingesters for kinodb. Reads robot trajectory data from
//! various formats and writes `.kdb` files.
//!
//! ## Supported formats
//!
//! - **HDF5** (robomimic / LIBERO / DROID style) — [`hdf5::Hdf5Ingester`]
//! - LeRobot — planned
//! - RLDS — planned

pub mod hdf5;
