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

use kinodb_core::{Episode, EpisodeId, EpisodeMeta, Frame, ImageObs, KdbWriter};

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
}

impl std::fmt::Display for Hdf5Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hdf5Error::Hdf5(e) => write!(f, "HDF5 error: {}", e),
            Hdf5Error::Write(e) => write!(f, "write error: {}", e),
            Hdf5Error::Io(e) => write!(f, "I/O error: {}", e),
            Hdf5Error::MissingGroup(g) => write!(f, "missing group: {}", g),
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
pub fn ingest_hdf5(
    hdf5_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &Hdf5IngestConfig,
) -> Result<IngestResult, Hdf5Error> {
    let hdf5_path = hdf5_path.as_ref();
    let output_path = output_path.as_ref();

    let file = hdf5::File::open(hdf5_path)?;

    let data_group = file
        .group("data")
        .map_err(|_| Hdf5Error::MissingGroup("data".to_string()))?;

    // Discover demo groups: sorted numerically
    let mut demo_names: Vec<String> = Vec::new();
    for name in data_group.member_names()? {
        if name.starts_with("demo_") {
            demo_names.push(name);
        }
    }
    demo_names.sort_by(|a, b| {
        let num_a = a.strip_prefix("demo_").and_then(|s| s.parse::<u64>().ok());
        let num_b = b.strip_prefix("demo_").and_then(|s| s.parse::<u64>().ok());
        num_a.cmp(&num_b)
    });

    if demo_names.is_empty() {
        return Err(Hdf5Error::MissingGroup(
            "data/demo_* (no demo groups found)".to_string(),
        ));
    }

    if let Some(max) = config.max_episodes {
        demo_names.truncate(max);
    }

    let task = config.task.clone().unwrap_or_else(|| {
        hdf5_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_task")
            .to_string()
    });

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
    let actions_ds = group
        .dataset("actions")
        .map_err(|_| Hdf5Error::MissingGroup(format!("data/{}/actions", demo_name)))?;
    let actions = actions_ds.read_2d::<f32>()?;
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

    let mut camera_keys: Vec<String> = Vec::new();
    let mut state_keys: Vec<String> = Vec::new();

    if let Some(ref obs) = obs_group {
        for name in obs.member_names().unwrap_or_default() {
            if let Ok(ds) = obs.dataset(&name) {
                let ndim = ds.ndim();
                if ndim == 4 {
                    camera_keys.push(name);
                } else if ndim == 2 {
                    state_keys.push(name);
                }
            }
        }
    }
    camera_keys.sort();
    state_keys.sort();

    // ── Read state vectors ──────────────────────────────────
    let mut state_arrays: Vec<ndarray::Array2<f32>> = Vec::new();
    if let Some(ref obs) = obs_group {
        for key in &state_keys {
            if let Ok(ds) = obs.dataset(key.as_str()) {
                if let Ok(arr) = ds.read_2d::<f32>() {
                    state_arrays.push(arr);
                }
            }
        }
    }

    // ── Read image data ─────────────────────────────────────
    // Read as dynamic-dim array, then extract flat bytes
    let mut camera_data: Vec<(String, Vec<u8>, usize, usize, usize, usize)> = Vec::new();
    if let Some(ref obs) = obs_group {
        for key in &camera_keys {
            if let Ok(ds) = obs.dataset(key.as_str()) {
                let shape = ds.shape();
                if shape.len() == 4 {
                    let n = shape[0];
                    let h = shape[1];
                    let w = shape[2];
                    let c = shape[3];
                    // read_dyn returns ArrayD<u8> — works for any dimensionality
                    if let Ok(arr) = ds.read_dyn::<u8>() {
                        // .iter() gives us row-major order, which is what we want
                        let flat: Vec<u8> = arr.iter().copied().collect();
                        camera_data.push((key.clone(), flat, n, h, w, c));
                    }
                }
            }
        }
    }

    // ── Build frames ────────────────────────────────────────
    let mut frames = Vec::with_capacity(num_frames);

    for t in 0..num_frames {
        // State: concat all state arrays for this timestep
        let mut state: Vec<f32> = Vec::new();
        for arr in &state_arrays {
            let row = arr.row(t);
            state.extend(row.iter());
        }

        // Action
        let action: Vec<f32> = actions.row(t).to_vec();

        // Images
        let mut images = Vec::with_capacity(camera_data.len());
        for (cam_name, flat_data, _n, h, w, c) in &camera_data {
            let frame_size = h * w * c;
            let offset = t * frame_size;
            let end = offset + frame_size;
            if end <= flat_data.len() {
                let data = flat_data[offset..end].to_vec();
                images.push(ImageObs {
                    camera: cam_name.clone(),
                    width: *w as u32,
                    height: *h as u32,
                    channels: *c as u8,
                    data,
                });
            }
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
    let success = rewards.as_ref().and_then(|r| r.last().map(|&v| v > 0.0));

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
