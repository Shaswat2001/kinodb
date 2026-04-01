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

use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyValueError, PyIndexError};
use numpy::PyArrayMethods;

use kinodb_core::{KdbReader, kql};

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
        let meta = self.reader.read_meta(position).map_err(|e| {
            PyIndexError::new_err(format!("{}", e))
        })?;

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
    ///   - "states": numpy float32 array, shape (num_frames, state_dim)
    ///   - "actions": numpy float32 array, shape (num_frames, action_dim)
    ///   - "rewards": numpy float32 array, shape (num_frames,)
    ///   - "is_terminal": list of bools
    ///   - "images": dict of camera_name -> numpy uint8 array, shape (num_frames, H, W, C)
    fn read_episode(&self, position: usize) -> PyResult<PyObject> {
        let episode = self.reader.read_episode(position).map_err(|e| {
            PyIndexError::new_err(format!("{}", e))
        })?;

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

            let num_frames = episode.frames.len();

            // Actions: flatten into contiguous buffer, return as 2D numpy array
            if num_frames > 0 {
                let action_dim = episode.frames[0].action.len();
                let mut action_buf: Vec<f32> = Vec::with_capacity(num_frames * action_dim);
                for frame in &episode.frames {
                    action_buf.extend_from_slice(&frame.action);
                }
                let action_array = numpy::PyArray2::from_vec2_bound(
                    py,
                    &episode.frames.iter().map(|f| f.action.clone()).collect::<Vec<_>>(),
                ).map_err(|e| PyValueError::new_err(format!("actions array: {}", e)))?;
                result.set_item("actions", action_array)?;

                // States: same approach
                let state_dim = episode.frames[0].state.len();
                let state_array = numpy::PyArray2::from_vec2_bound(
                    py,
                    &episode.frames.iter().map(|f| f.state.clone()).collect::<Vec<_>>(),
                ).map_err(|e| PyValueError::new_err(format!("states array: {}", e)))?;
                result.set_item("states", state_array)?;

                // Rewards: 1D numpy array
                let rewards: Vec<f32> = episode.frames.iter().map(|f| f.reward.unwrap_or(0.0)).collect();
                let reward_array = numpy::PyArray1::from_vec_bound(py, rewards);
                result.set_item("rewards", reward_array)?;
            } else {
                // Empty episode
                let empty_2d = numpy::PyArray2::<f32>::zeros_bound(py, [0, 0], false);
                result.set_item("actions", &empty_2d)?;
                result.set_item("states", &empty_2d)?;
                let empty_1d = numpy::PyArray1::<f32>::zeros_bound(py, [0], false);
                result.set_item("rewards", &empty_1d)?;
            }

            // Terminals (bool list — small, no need for numpy)
            let terminals: Vec<bool> = episode.frames.iter().map(|f| f.is_terminal).collect();
            result.set_item("is_terminal", terminals)?;

            // Images: group by camera, return as dict of numpy arrays
            // { "front": numpy uint8 (T, H, W, C), "wrist": numpy uint8 (T, H, W, C) }
            if num_frames > 0 && !episode.frames[0].images.is_empty() {
                let images_dict = pyo3::types::PyDict::new_bound(py);

                // Collect camera names from first frame
                let cameras: Vec<String> = episode.frames[0].images.iter()
                    .map(|img| img.camera.clone())
                    .collect();

                for cam_name in &cameras {
                    let mut buf: Vec<u8> = Vec::new();
                    let mut valid_frames = 0usize;
                    let mut frame_size = 0usize;
                    let mut dims = (0u32, 0u32, 0u8);

                    for frame in &episode.frames {
                        for img in &frame.images {
                            if &img.camera == cam_name {
                                if valid_frames == 0 {
                                    dims = (img.height, img.width, img.channels);
                                    frame_size = img.data.len();
                                }
                                buf.extend_from_slice(&img.data);
                                valid_frames += 1;
                                break;
                            }
                        }
                    }

                    let (h, w, c) = dims;
                    if valid_frames > 0 && frame_size > 0 && buf.len() == valid_frames * frame_size {
                        // Infer actual H, W from the per-frame data size
                        let channels = if c > 0 { c as usize } else { 3 };
                        let pixels = frame_size / channels;
                        let actual_h = if h > 0 { h as usize } else { (pixels as f64).sqrt() as usize };
                        let actual_w = if w > 0 { w as usize } else { pixels / actual_h };

                        if actual_h * actual_w * channels == frame_size {
                            let flat = numpy::PyArray1::from_vec_bound(py, buf);
                            let shape = [valid_frames, actual_h, actual_w, channels];
                            let array = flat.reshape(shape)
                                .map_err(|e| PyValueError::new_err(format!("image reshape: {}", e)))?;
                            images_dict.set_item(cam_name, array)?;
                        }
                    }
                }

                result.set_item("images", images_dict)?;
            } else {
                result.set_item("images", pyo3::types::PyDict::new_bound(py))?;
            }

            Ok(result.into())
        })
    }

    /// Read an episode's actions and states only (no images). Much faster.
    ///
    /// Returns a dict with "meta", "actions" (numpy), "states" (numpy),
    /// "rewards" (numpy), "is_terminal". No "images" key.
    fn read_episode_actions_only(&self, position: usize) -> PyResult<PyObject> {
        let episode = self.reader.read_episode_actions_only(position).map_err(|e| {
            PyIndexError::new_err(format!("{}", e))
        })?;

        Python::with_gil(|py| {
            let result = pyo3::types::PyDict::new_bound(py);

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

            let num_frames = episode.frames.len();
            if num_frames > 0 {
                let action_array = numpy::PyArray2::from_vec2_bound(
                    py,
                    &episode.frames.iter().map(|f| f.action.clone()).collect::<Vec<_>>(),
                ).map_err(|e| PyValueError::new_err(format!("actions: {}", e)))?;
                result.set_item("actions", action_array)?;

                let state_array = numpy::PyArray2::from_vec2_bound(
                    py,
                    &episode.frames.iter().map(|f| f.state.clone()).collect::<Vec<_>>(),
                ).map_err(|e| PyValueError::new_err(format!("states: {}", e)))?;
                result.set_item("states", state_array)?;

                let rewards: Vec<f32> = episode.frames.iter().map(|f| f.reward.unwrap_or(0.0)).collect();
                result.set_item("rewards", numpy::PyArray1::from_vec_bound(py, rewards))?;
            } else {
                result.set_item("actions", numpy::PyArray2::<f32>::zeros_bound(py, [0, 0], false))?;
                result.set_item("states", numpy::PyArray2::<f32>::zeros_bound(py, [0, 0], false))?;
                result.set_item("rewards", numpy::PyArray1::<f32>::zeros_bound(py, [0], false))?;
            }

            let terminals: Vec<bool> = episode.frames.iter().map(|f| f.is_terminal).collect();
            result.set_item("is_terminal", terminals)?;

            Ok(result.into())
        })
    }

    /// Filter episodes using a KQL query string.
    /// Returns a list of matching episode positions (ints).
    fn query(&self, query_str: &str) -> PyResult<Vec<usize>> {
        kql::filter_reader(&self.reader, query_str).map_err(|e| {
            PyValueError::new_err(format!("{}", e))
        })
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
    let reader = KdbReader::open(path).map_err(|e| {
        PyIOError::new_err(format!("failed to open '{}': {}", path, e))
    })?;
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