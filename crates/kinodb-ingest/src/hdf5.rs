//! Ingest robomimic / LIBERO style HDF5 files into `.kdb`.
//!
//! ## Expected HDF5 structure
//!
//! ```text
//! data/
//!   demo_0/
//!     actions          (N, action_dim) float32
//!     rewards          (N,) float32           [optional]
//!     dones            (N,) float32 or int    [optional]
//!     obs/
//!       agentview_image      (N, H, W, 3) uint8  [optional camera]
//!       robot0_eye_in_hand_image  (N, H, W, 3) uint8  [optional camera]
//!       robot0_eef_pos       (N, D) float32   [optional state]
//!       robot0_eef_quat      (N, D) float32   [optional state]
//!       robot0_gripper_qpos  (N, D) float32   [optional state]
//!       ...any other keys...
//!   demo_1/
//!     ...
//! ```
//!
//! The ingester auto-discovers:
//! - Demo groups (anything matching `demo_*` under `data/`)
//! - Camera keys (any obs dataset with ndim=4 and dtype=uint8)
//! - State keys (any obs dataset with ndim=2 and dtype=float)

use std::path::Path;

use kinodb_core::{
    Episode, EpisodeId, EpisodeMeta, Frame, ImageObs, KdbWriter,
};

/// Configuration for HDF5 ingestion.
#[derive(Debug, Clone)]
pub struct Hdf5IngestConfig {
    /// Name of the robot embodiment (e.g. "franka", "widowx").
    /// HDF5 files don't store this, so the user must provide it.
    pub embodiment: String,

    /// Task description. If None, we try to read it from the HDF5
    /// attributes, otherwise default to the filename.
    pub task: Option<String>,

    /// Control frequency in Hz. HDF5 files rarely store this.
    pub fps: f32,

    /// If set, only ingest the first N episodes.
    pub max_episodes: Option<usize>,
}

impl Default for Hdf5IngestConfig {
    fn default() -> Self {
        Self {
            embodiment: "unknown".to_string(),
            task: None,
            fps: 10.0,
            max_episodes: None,
        }
    }
}

/// Errors from HDF5 ingestion.
#[derive(Debug)]
pub enum Hdf5Error {
    Hdf5(hdf5::Error),
    Write(kinodb_core::WriteError),
    Io(std::io::Error),
    /// The HDF5 file doesn't have the expected structure.
    MissingGroup(String),
    /// A dataset has an unexpected shape.
    UnexpectedShape {
        dataset: String,
        expected: String,
        got: String,
    },
}

impl std::fmt::Display for Hdf5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hdf5Error::Hdf5(e) => write!(f, "HDF5 error: {}", e),
            Hdf5Error::Write(e) => write!(f, "write error: {}", e),
            Hdf5Error::Io(e) => write!(f, "I/O error: {}", e),
            Hdf5Error::MissingGroup(g) => write!(f, "missing group: {}", g),
            Hdf5Error::UnexpectedShape {
                dataset,
                expected,
                got,
            } => write!(
                f,
                "dataset '{}': expected shape {}, got {}",
                dataset, expected, got
            ),
        }
    }
}

impl std::error::Error for Hdf5Error {}

impl From<hdf5::Error> for Hdf5Error {
    fn from(e: hdf5::Error) -> Self {
        Hdf5Error::Hdf5(e)
    }
}

impl From<kinodb_core::WriteError> for Hdf5Error {
    fn from(e: kinodb_core::WriteError) -> Self {
        Hdf5Error::Write(e)
    }
}

impl From<std::io::Error> for Hdf5Error {
    fn from(e: std::io::Error) -> Self {
        Hdf5Error::Io(e)
    }
}

/// Result type for HDF5 ingestion.
pub struct IngestResult {
    pub num_episodes: usize,
    pub total_frames: u64,
    pub output_path: String,
}

/// Ingest a robomimic/LIBERO-style HDF5 file into a `.kdb` file.
///
/// Returns the number of episodes and frames written.
pub fn ingest_hdf5(
    hdf5_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &Hdf5IngestConfig,
) -> Result<IngestResult, Hdf5Error> {
    let hdf5_path = hdf5_path.as_ref();
    let output_path = output_path.as_ref();

    let file = hdf5::File::open(hdf5_path)?;

    // Find the data group
    let data_group = file.group("data").map_err(|_| {
        Hdf5Error::MissingGroup("data".to_string())
    })?;

    // Discover demo groups: sorted by name so order is deterministic
    let mut demo_names: Vec<String> = Vec::new();
    for name in data_group.member_names()? {
        if name.starts_with("demo_") {
            demo_names.push(name);
        }
    }
    demo_names.sort_by(|a, b| {
        // Sort numerically: demo_0, demo_1, ..., demo_10, demo_11
        let num_a = a.strip_prefix("demo_").and_then(|s| s.parse::<u64>().ok());
        let num_b = b.strip_prefix("demo_").and_then(|s| s.parse::<u64>().ok());
        num_a.cmp(&num_b)
    });

    if demo_names.is_empty() {
        return Err(Hdf5Error::MissingGroup(
            "data/demo_* (no demo groups found)".to_string(),
        ));
    }

    // Apply max_episodes limit
    if let Some(max) = config.max_episodes {
        demo_names.truncate(max);
    }

    // Determine task string
    let task = config.task.clone().unwrap_or_else(|| {
        // Try to get from HDF5 attributes, fall back to filename
        hdf5_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_task")
            .to_string()
    });

    // Create writer
    let mut writer = KdbWriter::create(output_path)?;
    let mut total_frames: u64 = 0;

    for (idx, demo_name) in demo_names.iter().enumerate() {
        let demo_group = data_group.group(demo_name)?;

        let episode = read_demo_group(
            &demo_group,
            demo_name,
            idx,
            &config.embodiment,
            &task,
            config.fps,
        )?;

        total_frames += episode.frames.len() as u64;
        writer.write_episode(&episode)?;
    }

    writer.finish()?;

    Ok(IngestResult {
        num_episodes: demo_names.len(),
        total_frames,
        output_path: output_path.to_string_lossy().to_string(),
    })
}

/// Read one demo group into an Episode.
fn read_demo_group(
    group: &hdf5::Group,
    demo_name: &str,
    idx: usize,
    embodiment: &str,
    task: &str,
    fps: f32,
) -> Result<Episode, Hdf5Error> {
    // ── Read actions (required) ─────────────────────────────
    let actions_ds = group.dataset("actions").map_err(|_| {
        Hdf5Error::MissingGroup(format!("data/{}/actions", demo_name))
    })?;
    let actions: ndarray::Array2<f32> = actions_ds.read_2d()?;
    let num_frames = actions.nrows();
    let action_dim = actions.ncols();

    // ── Read rewards (optional) ─────────────────────────────
    let rewards: Option<Vec<f32>> = group
        .dataset("rewards")
        .ok()
        .and_then(|ds| ds.read_1d::<f32>().ok())
        .map(|arr| arr.to_vec());

    // ── Read dones (optional) ───────────────────────────────
    let dones: Option<Vec<f32>> = group
        .dataset("dones")
        .ok()
        .and_then(|ds| ds.read_1d::<f32>().ok())
        .map(|arr| arr.to_vec());

    // ── Discover observations ───────────────────────────────
    let obs_group = group.group("obs").ok();

    // Classify obs keys into cameras (4D uint8) and state (2D float)
    let mut camera_keys: Vec<String> = Vec::new();
    let mut state_keys: Vec<String> = Vec::new();

    if let Some(ref obs) = obs_group {
        for name in obs.member_names().unwrap_or_default() {
            if let Ok(ds) = obs.dataset(&name) {
                let shape = ds.shape();
                let ndim = shape.len();

                if ndim == 4 {
                    // Likely an image: (N, H, W, C)
                    camera_keys.push(name);
                } else if ndim == 2 {
                    // Likely state: (N, D)
                    state_keys.push(name);
                }
                // Skip 1D or other shapes
            }
        }
    }
    camera_keys.sort();
    state_keys.sort();

    // ── Read state vectors ──────────────────────────────────
    // Concatenate all state keys into one vector per frame
    let mut state_arrays: Vec<ndarray::Array2<f32>> = Vec::new();
    if let Some(ref obs) = obs_group {
        for key in &state_keys {
            if let Ok(ds) = obs.dataset(key) {
                if let Ok(arr) = ds.read_2d::<f32>() {
                    state_arrays.push(arr);
                }
            }
        }
    }

    let state_dim: usize = state_arrays.iter().map(|a| a.ncols()).sum();

    // ── Read image data ─────────────────────────────────────
    // We read all camera data upfront: Vec<(camera_name, Array4<u8>)>
    let mut camera_data: Vec<(String, ndarray::Array4<u8>)> = Vec::new();
    if let Some(ref obs) = obs_group {
        for key in &camera_keys {
            if let Ok(ds) = obs.dataset(key) {
                // Read as 4D: (N, H, W, C)
                if let Ok(arr) = ds.read::<u8, ndarray::Ix4>() {
                    camera_data.push((key.clone(), arr));
                }
            }
        }
    }

    // ── Build frames ────────────────────────────────────────
    let mut frames = Vec::with_capacity(num_frames);

    for t in 0..num_frames {
        // State: concat all state arrays for this timestep
        let mut state = Vec::with_capacity(state_dim);
        for arr in &state_arrays {
            let row = arr.row(t);
            state.extend(row.iter());
        }

        // Action
        let action: Vec<f32> = actions.row(t).to_vec();

        // Images
        let mut images = Vec::with_capacity(camera_data.len());
        for (cam_name, arr) in &camera_data {
            let frame_img = arr.index_axis(ndarray::Axis(0), t);
            let height = frame_img.shape()[0] as u32;
            let width = frame_img.shape()[1] as u32;
            let channels = frame_img.shape()[2] as u8;

            // ndarray stores in row-major, which is what we want (H, W, C)
            let data: Vec<u8> = frame_img.iter().copied().collect();

            images.push(ImageObs {
                camera: cam_name.clone(),
                width,
                height,
                channels,
                data,
            });
        }

        // Reward
        let reward = rewards.as_ref().and_then(|r| r.get(t).copied());

        // Terminal
        let is_terminal = if let Some(ref d) = dones {
            d.get(t).map(|&v| v > 0.5).unwrap_or(t == num_frames - 1)
        } else {
            t == num_frames - 1
        };

        frames.push(Frame {
            timestep: t as u32,
            images,
            state,
            action,
            reward,
            is_terminal,
        });
    }

    // ── Determine success ───────────────────────────────────
    // robomimic convention: last reward > 0 means success
    let success = rewards
        .as_ref()
        .and_then(|r| r.last().map(|&v| v > 0.0));

    let total_reward = rewards.as_ref().map(|r| r.iter().sum());

    let meta = EpisodeMeta {
        id: EpisodeId(idx as u64),
        embodiment: embodiment.to_string(),
        language_instruction: task.to_string(),
        num_frames: num_frames as u32,
        fps,
        action_dim: action_dim as u16,
        success,
        total_reward,
    };

    Ok(Episode { meta, frames })
}

// We also bring in ndarray since hdf5 crate uses it
use hdf5;

// Re-export ndarray types used by hdf5
extern crate ndarray;
