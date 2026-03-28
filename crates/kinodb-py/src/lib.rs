//! Python bindings for kinodb.
//!
//! Exposes the core reader, KQL filtering, and episode data as
//! native Python objects. Install with maturin:
//!
//! ```bash
//! cd crates/kinodb-py
//! pip install maturin
//! maturin develop
//! ```
//!
//! Then in Python:
//! ```python
//! import kinodb
//!
//! db = kinodb.open("data.kdb")
//! print(db.num_episodes())
//!
//! meta = db.read_meta(0)
//! print(meta["embodiment"], meta["task"])
//!
//! ep = db.read_episode(0)
//! print(ep["actions"])    # list of list of floats
//! print(ep["images"][0])  # dict with camera, width, height, data (bytes)
//!
//! # KQL filtering
//! hits = db.query("embodiment = 'franka' AND success = true")
//! print(f"{len(hits)} matching episodes")
//! ```

use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;

use kinodb_core::{kql, KdbReader};

/// A kinodb database reader, exposed to Python.
#[pyclass(name = "Database")]
struct PyDatabase {
    reader: KdbReader,
    path: String,
}

#[pymethods]
impl PyDatabase {
    /// Number of episodes in this database.
    fn num_episodes(&self) -> usize {
        self.reader.num_episodes()
    }

    /// Total frames across all episodes.
    fn num_frames(&self) -> u64 {
        self.reader.num_frames()
    }

    /// Database file path.
    fn path(&self) -> &str {
        &self.path
    }

    /// Format version as a string like "0.1".
    fn version(&self) -> String {
        let h = self.reader.header();
        format!("{}.{}", h.version_major, h.version_minor)
    }

    /// Read metadata for one episode (by position). Returns a dict.
    fn read_meta(&self, position: usize) -> PyResult<PyObject> {
        let meta = self
            .reader
            .read_meta(position)
            .map_err(|e| PyIndexError::new_err(format!("{}", e)))?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new_bound(py);
            dict.set_item("episode_id", meta.id.0)?;
            dict.set_item("embodiment", &meta.embodiment)?;
            dict.set_item("task", &meta.language_instruction)?;
            dict.set_item("num_frames", meta.num_frames)?;
            dict.set_item("fps", meta.fps)?;
            dict.set_item("action_dim", meta.action_dim)?;
            dict.set_item("success", meta.success)?;
            dict.set_item("total_reward", meta.total_reward)?;
            Ok(dict.into())
        })
    }

    /// Read a full episode (meta + all frames) by position. Returns a dict.
    ///
    /// The dict contains:
    ///   - "meta": dict (same as read_meta)
    ///   - "states": list of list of floats (one per frame)
    ///   - "actions": list of list of floats (one per frame)
    ///   - "rewards": list of floats
    ///   - "is_terminal": list of bools
    ///   - "images": list of dicts per frame, each with camera entries
    fn read_episode(&self, position: usize) -> PyResult<PyObject> {
        let episode = self
            .reader
            .read_episode(position)
            .map_err(|e| PyIndexError::new_err(format!("{}", e)))?;

        Python::with_gil(|py| {
            let result = pyo3::types::PyDict::new_bound(py);

            // Meta
            let meta_dict = pyo3::types::PyDict::new_bound(py);
            meta_dict.set_item("episode_id", episode.meta.id.0)?;
            meta_dict.set_item("embodiment", &episode.meta.embodiment)?;
            meta_dict.set_item("task", &episode.meta.language_instruction)?;
            meta_dict.set_item("num_frames", episode.meta.num_frames)?;
            meta_dict.set_item("fps", episode.meta.fps)?;
            meta_dict.set_item("action_dim", episode.meta.action_dim)?;
            meta_dict.set_item("success", episode.meta.success)?;
            meta_dict.set_item("total_reward", episode.meta.total_reward)?;
            result.set_item("meta", meta_dict)?;

            // States
            let states: Vec<Vec<f32>> = episode.frames.iter().map(|f| f.state.clone()).collect();
            result.set_item("states", states)?;

            // Actions
            let actions: Vec<Vec<f32>> = episode.frames.iter().map(|f| f.action.clone()).collect();
            result.set_item("actions", actions)?;

            // Rewards
            let rewards: Vec<f32> = episode
                .frames
                .iter()
                .map(|f| f.reward.unwrap_or(0.0))
                .collect();
            result.set_item("rewards", rewards)?;

            // Terminals
            let terminals: Vec<bool> = episode.frames.iter().map(|f| f.is_terminal).collect();
            result.set_item("is_terminal", terminals)?;

            // Images: list of list of dicts
            // images[frame_idx] = [{"camera": str, "width": int, ...}, ...]
            let mut all_frame_images: Vec<PyObject> = Vec::new();
            for frame in &episode.frames {
                let frame_imgs = pyo3::types::PyList::empty_bound(py);
                for img in &frame.images {
                    let img_dict = pyo3::types::PyDict::new_bound(py);
                    img_dict.set_item("camera", &img.camera)?;
                    img_dict.set_item("width", img.width)?;
                    img_dict.set_item("height", img.height)?;
                    img_dict.set_item("channels", img.channels)?;
                    img_dict.set_item("data", pyo3::types::PyBytes::new_bound(py, &img.data))?;
                    frame_imgs.append(img_dict)?;
                }
                all_frame_images.push(frame_imgs.into());
            }
            result.set_item("images", all_frame_images)?;

            Ok(result.into())
        })
    }

    /// Filter episodes using a KQL query string.
    /// Returns a list of matching episode positions (ints).
    fn query(&self, query_str: &str) -> PyResult<Vec<usize>> {
        kql::filter_reader(&self.reader, query_str)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))
    }

    /// Get a summary string (like `kino info`).
    fn summary(&self) -> String {
        let h = self.reader.header();
        let mut s = format!(
            "kinodb v{}.{}\n  Episodes: {}\n  Frames: {}",
            h.version_major, h.version_minor, h.num_episodes, h.num_frames,
        );
        if h.num_episodes > 0 {
            let avg = h.num_frames as f64 / h.num_episodes as f64;
            s.push_str(&format!("\n  Avg length: {:.1} frames/episode", avg));
        }
        s
    }

    fn __repr__(&self) -> String {
        format!(
            "kinodb.Database('{}', episodes={}, frames={})",
            self.path,
            self.reader.num_episodes(),
            self.reader.num_frames(),
        )
    }

    fn __len__(&self) -> usize {
        self.reader.num_episodes()
    }
}

/// Open a .kdb database file. Returns a Database object.
#[pyfunction]
fn open(path: &str) -> PyResult<PyDatabase> {
    let reader = KdbReader::open(path)
        .map_err(|e| PyIOError::new_err(format!("failed to open '{}': {}", path, e)))?;
    Ok(PyDatabase {
        reader,
        path: path.to_string(),
    })
}

/// The kinodb Python module.
#[pymodule]
fn kinodb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<PyDatabase>()?;
    Ok(())
}
