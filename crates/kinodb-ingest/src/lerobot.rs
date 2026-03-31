//! Ingest LeRobot v2/v3 datasets into `.kdb`.
//!
//! ## Expected directory structure (v3)
//!
//! ```text
//! dataset/
//!   meta/
//!     info.json              ← schema: features, fps, robot_type
//!     tasks.jsonl            ← task descriptions (v2) or tasks.parquet (v3)
//!     episodes/chunk-000/    ← episode metadata parquet (v3)
//!   data/
//!     chunk-000/
//!       file-000.parquet     ← actions, states, episode_index, timestamps
//!   videos/                  ← MP4 files (skipped for now)
//! ```
//!
//! ## v2 structure
//!
//! ```text
//! dataset/
//!   meta/
//!     info.json
//!     tasks.jsonl
//!     episodes.jsonl
//!   data/
//!     chunk-000/
//!       episode_000000.parquet
//!   videos/...
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use arrow::array::Array;

use kinodb_core::{Episode, EpisodeId, EpisodeMeta, Frame, KdbWriter};

/// Configuration for LeRobot ingestion.
#[derive(Debug, Clone)]
pub struct LeRobotIngestConfig {
    /// Override embodiment name (if not in info.json).
    pub embodiment: Option<String>,
    /// Override task (if not in tasks.jsonl).
    pub task: Option<String>,
    /// Max episodes to ingest.
    pub max_episodes: Option<usize>,
}

impl Default for LeRobotIngestConfig {
    fn default() -> Self {
        Self {
            embodiment: None,
            task: None,
            max_episodes: None,
        }
    }
}

/// Errors from LeRobot ingestion.
#[derive(Debug)]
pub enum LeRobotError {
    Io(std::io::Error),
    Write(kinodb_core::WriteError),
    Parquet(parquet::errors::ParquetError),
    Arrow(arrow::error::ArrowError),
    Json(serde_json::Error),
    Missing(String),
    Format(String),
}

impl std::fmt::Display for LeRobotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeRobotError::Io(e) => write!(f, "I/O error: {}", e),
            LeRobotError::Write(e) => write!(f, "write error: {}", e),
            LeRobotError::Parquet(e) => write!(f, "parquet error: {}", e),
            LeRobotError::Arrow(e) => write!(f, "arrow error: {}", e),
            LeRobotError::Json(e) => write!(f, "JSON error: {}", e),
            LeRobotError::Missing(s) => write!(f, "missing: {}", s),
            LeRobotError::Format(s) => write!(f, "format error: {}", s),
        }
    }
}

impl std::error::Error for LeRobotError {}

impl From<std::io::Error> for LeRobotError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<kinodb_core::WriteError> for LeRobotError {
    fn from(e: kinodb_core::WriteError) -> Self {
        Self::Write(e)
    }
}
impl From<parquet::errors::ParquetError> for LeRobotError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        Self::Parquet(e)
    }
}
impl From<arrow::error::ArrowError> for LeRobotError {
    fn from(e: arrow::error::ArrowError) -> Self {
        Self::Arrow(e)
    }
}
impl From<serde_json::Error> for LeRobotError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub struct IngestResult {
    pub num_episodes: usize,
    pub total_frames: u64,
    pub output_path: String,
}

/// Ingest a LeRobot v2 or v3 dataset directory into a `.kdb` file.
pub fn ingest_lerobot(
    dataset_dir: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &LeRobotIngestConfig,
) -> Result<IngestResult, LeRobotError> {
    let dataset_dir = dataset_dir.as_ref();
    let output_path = output_path.as_ref();

    // ── Read info.json ──────────────────────────────────────
    let info_path = dataset_dir.join("meta").join("info.json");
    if !info_path.exists() {
        return Err(LeRobotError::Missing(format!(
            "meta/info.json not found in {}",
            dataset_dir.display()
        )));
    }
    let info_str = fs::read_to_string(&info_path)?;
    let info: serde_json::Value = serde_json::from_str(&info_str)?;

    let fps = info.get("fps").and_then(|v| v.as_f64()).unwrap_or(30.0) as f32;

    let robot_type = info
        .get("robot_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let embodiment = config
        .embodiment
        .clone()
        .unwrap_or_else(|| robot_type.to_string());

    // ── Read tasks ──────────────────────────────────────────
    let tasks = read_tasks(dataset_dir)?;

    // ── Discover and read parquet data files ─────────────────
    let data_dir = dataset_dir.join("data");
    if !data_dir.exists() {
        return Err(LeRobotError::Missing(format!(
            "data/ directory not found in {}",
            dataset_dir.display()
        )));
    }

    let mut parquet_files = find_parquet_files(&data_dir);
    parquet_files.sort();

    if parquet_files.is_empty() {
        return Err(LeRobotError::Missing(
            "no parquet files found in data/".to_string(),
        ));
    }

    // ── Read all rows, group by episode_index ───────────────
    let mut episodes_map: BTreeMap<i64, Vec<RowData>> = BTreeMap::new();

    for pq_path in &parquet_files {
        read_parquet_into_episodes(pq_path, &mut episodes_map)?;
    }

    if episodes_map.is_empty() {
        return Err(LeRobotError::Format(
            "no episode data found in parquet files".to_string(),
        ));
    }

    // ── Apply limit ─────────────────────────────────────────
    let episode_indices: Vec<i64> = episodes_map.keys().copied().collect();
    let n_episodes = match config.max_episodes {
        Some(max) => std::cmp::min(max, episode_indices.len()),
        None => episode_indices.len(),
    };

    // ── Write to .kdb ───────────────────────────────────────
    let mut writer = KdbWriter::create(output_path)?;
    let mut total_frames: u64 = 0;

    for (write_idx, &ep_idx) in episode_indices.iter().take(n_episodes).enumerate() {
        let rows = episodes_map.get(&ep_idx).unwrap();

        let task_str = config.task.clone().unwrap_or_else(|| {
            // Try to get task from task_index in the first row
            rows.first()
                .and_then(|r| r.task_index)
                .and_then(|ti| tasks.get(&ti).cloned())
                .unwrap_or_else(|| format!("episode_{}", ep_idx))
        });

        let num_frames = rows.len() as u32;
        let action_dim = rows.first().map(|r| r.action.len()).unwrap_or(0) as u16;

        let meta = EpisodeMeta {
            id: EpisodeId(write_idx as u64),
            embodiment: embodiment.clone(),
            language_instruction: task_str,
            num_frames,
            fps,
            action_dim,
            success: None,
            total_reward: None,
        };

        let frames: Vec<Frame> = rows
            .iter()
            .enumerate()
            .map(|(t, row)| Frame {
                timestep: t as u32,
                images: vec![], // video decoding not yet implemented
                state: row.state.clone(),
                action: row.action.clone(),
                reward: None,
                is_terminal: t == rows.len() - 1,
            })
            .collect();

        let episode = Episode { meta, frames };
        writer.write_episode(&episode)?;
        total_frames += num_frames as u64;
    }

    writer.finish()?;

    Ok(IngestResult {
        num_episodes: n_episodes,
        total_frames,
        output_path: output_path.to_string_lossy().to_string(),
    })
}

/// One row of data from a parquet file.
struct RowData {
    action: Vec<f32>,
    state: Vec<f32>,
    task_index: Option<i64>,
}

/// Read tasks from tasks.jsonl (v2) or tasks.parquet (v3).
fn read_tasks(dataset_dir: &Path) -> Result<BTreeMap<i64, String>, LeRobotError> {
    let mut tasks: BTreeMap<i64, String> = BTreeMap::new();

    // Try tasks.jsonl first (v2 format)
    let jsonl_path = dataset_dir.join("meta").join("tasks.jsonl");
    if jsonl_path.exists() {
        let content = fs::read_to_string(&jsonl_path)?;
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
                let idx = val.get("task_index").and_then(|v| v.as_i64()).unwrap_or(-1);
                let desc = val.get("task").and_then(|v| v.as_str()).unwrap_or("");
                if idx >= 0 && !desc.is_empty() {
                    tasks.insert(idx, desc.to_string());
                }
            }
        }
        return Ok(tasks);
    }

    // Try tasks.parquet (v3 format)
    let parquet_path = dataset_dir.join("meta").join("tasks.parquet");
    if parquet_path.exists() {
        let file = fs::File::open(&parquet_path)?;
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReader::try_new(file, 1024)?;
        for batch in reader {
            let batch = batch?;
            // Look for task_index and task columns
            let schema = batch.schema();
            let idx_col = schema.index_of("task_index").ok();
            let task_col = schema.index_of("task").ok();

            if let (Some(ic), Some(tc)) = (idx_col, task_col) {
                let indices = batch
                    .column(ic)
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>();
                let descriptions = batch
                    .column(tc)
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>();

                if let (Some(idx_arr), Some(desc_arr)) = (indices, descriptions) {
                    for row in 0..batch.num_rows() {
                        if !idx_arr.is_null(row) && !desc_arr.is_null(row) {
                            tasks.insert(idx_arr.value(row), desc_arr.value(row).to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(tasks)
}

/// Read a parquet file and group rows into the episodes map.
fn read_parquet_into_episodes(
    path: &Path,
    episodes: &mut BTreeMap<i64, Vec<RowData>>,
) -> Result<(), LeRobotError> {
    let file = fs::File::open(path)?;
    let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReader::try_new(file, 4096)?;

    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();
        let num_rows = batch.num_rows();

        // Find episode_index column
        let ep_col_idx = schema.index_of("episode_index").map_err(|_| {
            LeRobotError::Format(format!("no 'episode_index' column in {}", path.display()))
        })?;

        let ep_indices = batch
            .column(ep_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .ok_or_else(|| LeRobotError::Format("episode_index column is not Int64".to_string()))?;

        // Find task_index column (optional)
        let task_col_idx = schema.index_of("task_index").ok();

        // Discover action columns: anything starting with "action"
        // and state columns: anything starting with "observation.state"
        let mut action_cols: Vec<(usize, String)> = Vec::new();
        let mut state_cols: Vec<(usize, String)> = Vec::new();

        for (i, field) in schema.fields().iter().enumerate() {
            let name = field.name();
            if name == "action" || name.starts_with("action.") {
                action_cols.push((i, name.clone()));
            } else if name == "observation.state" || name.starts_with("observation.state.") {
                state_cols.push((i, name.clone()));
            }
        }
        action_cols.sort_by(|a, b| a.1.cmp(&b.1));
        state_cols.sort_by(|a, b| a.1.cmp(&b.1));

        // Process each row
        for row in 0..num_rows {
            let ep_idx = ep_indices.value(row);

            // Extract action values
            let action = extract_f32_row(&batch, &action_cols, row);

            // Extract state values
            let state = extract_f32_row(&batch, &state_cols, row);

            // Task index
            let task_index = task_col_idx.and_then(|ci| {
                batch
                    .column(ci)
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .map(|arr| arr.value(row))
            });

            episodes
                .entry(ep_idx)
                .or_insert_with(Vec::new)
                .push(RowData {
                    action,
                    state,
                    task_index,
                });
        }
    }

    Ok(())
}

/// Extract f32 values from columns for a given row.
/// Handles both single-column lists and multi-column scalar floats.
fn extract_f32_row(
    batch: &arrow::record_batch::RecordBatch,
    cols: &[(usize, String)],
    row: usize,
) -> Vec<f32> {
    let mut values = Vec::new();

    for (col_idx, _name) in cols {
        let col = batch.column(*col_idx);

        // Try as Float32Array (scalar column)
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float32Array>() {
            if !arr.is_null(row) {
                values.push(arr.value(row));
            }
            continue;
        }

        // Try as Float64Array and convert
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
            if !arr.is_null(row) {
                values.push(arr.value(row) as f32);
            }
            continue;
        }

        // Try as FixedSizeListArray (e.g. action is a list of floats)
        if let Some(list_arr) = col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
        {
            let inner = list_arr.value(row);
            if let Some(f32_arr) = inner.as_any().downcast_ref::<arrow::array::Float32Array>() {
                for i in 0..f32_arr.len() {
                    if !f32_arr.is_null(i) {
                        values.push(f32_arr.value(i));
                    }
                }
            } else if let Some(f64_arr) =
                inner.as_any().downcast_ref::<arrow::array::Float64Array>()
            {
                for i in 0..f64_arr.len() {
                    if !f64_arr.is_null(i) {
                        values.push(f64_arr.value(i) as f32);
                    }
                }
            }
            continue;
        }

        // Try as regular ListArray
        if let Some(list_arr) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
            let inner = list_arr.value(row);
            if let Some(f32_arr) = inner.as_any().downcast_ref::<arrow::array::Float32Array>() {
                for i in 0..f32_arr.len() {
                    if !f32_arr.is_null(i) {
                        values.push(f32_arr.value(i));
                    }
                }
            } else if let Some(f64_arr) =
                inner.as_any().downcast_ref::<arrow::array::Float64Array>()
            {
                for i in 0..f64_arr.len() {
                    if !f64_arr.is_null(i) {
                        values.push(f64_arr.value(i) as f32);
                    }
                }
            }
            continue;
        }
    }

    values
}

/// Recursively find all .parquet files under a directory.
fn find_parquet_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_parquet_files(&path));
            } else if path.extension().map(|e| e == "parquet").unwrap_or(false) {
                files.push(path);
            }
        }
    }
    files
}
