//! Ingest RLDS datasets (TFRecord format) into `.kdb`.
//!
//! Parses TFRecord files directly — **no TensorFlow dependency**.
//!
//! ## Expected structure
//!
//! ```text
//! dataset/
//!   1.0.0/
//!     dataset_info.json           ← schema info (optional)
//!     train.tfrecord-00000-of-00005
//!     train.tfrecord-00001-of-00005
//!     ...
//! ```
//!
//! Each TFRecord file contains serialized `tf.train.Example` protos.
//! RLDS stores one step per Example, with fields:
//! - `is_first`, `is_last`, `is_terminal` (bools)
//! - `observation/*` (nested float/int/bytes features)
//! - `action` (float list)
//! - `reward` (float)
//! - `discount` (float, ignored)

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

use kinodb_core::{Episode, EpisodeId, EpisodeMeta, Frame, ImageObs, KdbWriter};

/// Configuration for RLDS ingestion.
#[derive(Debug, Clone)]
pub struct RldsIngestConfig {
    /// Override embodiment name.
    pub embodiment: String,
    /// Override task description.
    pub task: Option<String>,
    /// Control frequency in Hz.
    pub fps: f32,
    /// Max episodes to ingest.
    pub max_episodes: Option<usize>,
}

impl Default for RldsIngestConfig {
    fn default() -> Self {
        Self {
            embodiment: "unknown".to_string(),
            task: None,
            fps: 10.0,
            max_episodes: None,
        }
    }
}

/// Errors from RLDS ingestion.
#[derive(Debug)]
pub enum RldsError {
    Io(io::Error),
    Write(kinodb_core::WriteError),
    Format(String),
    Crc(String),
}

impl std::fmt::Display for RldsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RldsError::Io(e) => write!(f, "I/O error: {}", e),
            RldsError::Write(e) => write!(f, "write error: {}", e),
            RldsError::Format(s) => write!(f, "format error: {}", s),
            RldsError::Crc(s) => write!(f, "CRC error: {}", s),
        }
    }
}

impl std::error::Error for RldsError {}

impl From<io::Error> for RldsError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<kinodb_core::WriteError> for RldsError {
    fn from(e: kinodb_core::WriteError) -> Self {
        Self::Write(e)
    }
}

pub struct IngestResult {
    pub num_episodes: usize,
    pub total_frames: u64,
    pub output_path: String,
}

/// Ingest RLDS TFRecord files into a `.kdb` file.
pub fn ingest_rlds(
    dataset_dir: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &RldsIngestConfig,
) -> Result<IngestResult, RldsError> {
    let dataset_dir = dataset_dir.as_ref();
    let output_path = output_path.as_ref();

    // Find all TFRecord files
    let mut tfrecord_files = find_tfrecord_files(dataset_dir);
    tfrecord_files.sort();

    if tfrecord_files.is_empty() {
        return Err(RldsError::Format(format!(
            "no TFRecord files found in {}",
            dataset_dir.display()
        )));
    }

    // Try to read language instruction from dataset_info.json
    let default_task = config
        .task
        .clone()
        .unwrap_or_else(|| read_dataset_task(dataset_dir).unwrap_or_else(|| "unknown".to_string()));

    // Read all steps from all TFRecord files
    let mut all_steps: Vec<RldsStep> = Vec::new();
    for path in &tfrecord_files {
        let steps = read_tfrecord_file(path)?;
        all_steps.extend(steps);
    }

    if all_steps.is_empty() {
        return Err(RldsError::Format(
            "no steps found in TFRecord files".to_string(),
        ));
    }

    // Group steps into episodes using is_first / is_last flags
    let episodes = group_into_episodes(&all_steps);

    if episodes.is_empty() {
        return Err(RldsError::Format("no complete episodes found".to_string()));
    }

    // Apply limit
    let n_episodes = match config.max_episodes {
        Some(max) => std::cmp::min(max, episodes.len()),
        None => episodes.len(),
    };

    // Write to .kdb
    let mut writer = KdbWriter::create(output_path)?;
    let mut total_frames: u64 = 0;

    for (ep_idx, ep_steps) in episodes.iter().take(n_episodes).enumerate() {
        let num_frames = ep_steps.len() as u32;
        let action_dim = ep_steps.first().map(|s| s.action.len()).unwrap_or(0) as u16;

        let task_str = ep_steps
            .first()
            .and_then(|s| s.language_instruction.clone())
            .unwrap_or_else(|| default_task.clone());

        let meta = EpisodeMeta {
            id: EpisodeId(ep_idx as u64),
            embodiment: config.embodiment.clone(),
            language_instruction: task_str,
            num_frames,
            fps: config.fps,
            action_dim,
            success: ep_steps.last().map(|s| s.is_terminal),
            total_reward: Some(ep_steps.iter().map(|s| s.reward).sum()),
        };

        let frames: Vec<Frame> = ep_steps
            .iter()
            .enumerate()
            .map(|(t, step)| {
                let images = step
                    .images
                    .iter()
                    .map(|(name, img_data, w, h)| ImageObs {
                        camera: name.clone(),
                        width: *w,
                        height: *h,
                        channels: 3,
                        data: img_data.clone(),
                    })
                    .collect();

                Frame {
                    timestep: t as u32,
                    images,
                    state: step.state.clone(),
                    action: step.action.clone(),
                    reward: Some(step.reward),
                    is_terminal: step.is_terminal || (t == ep_steps.len() - 1),
                }
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

// ── Internal types ──────────────────────────────────────────

/// One parsed RLDS step.
struct RldsStep {
    is_first: bool,
    is_last: bool,
    is_terminal: bool,
    action: Vec<f32>,
    state: Vec<f32>,
    reward: f32,
    language_instruction: Option<String>,
    /// (camera_name, raw_rgb_bytes, width, height)
    images: Vec<(String, Vec<u8>, u32, u32)>,
}

// ── TFRecord binary parsing ─────────────────────────────────
//
// TFRecord format (per record):
//   [u64 LE: data_length]
//   [u32 LE: masked_crc32c of data_length bytes]
//   [bytes: data (data_length bytes)]
//   [u32 LE: masked_crc32c of data]

fn read_tfrecord_file(path: &Path) -> Result<Vec<RldsStep>, RldsError> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut steps = Vec::new();

    loop {
        // Read data length (u64 LE)
        let mut len_buf = [0u8; 8];
        match reader.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break, // end of file
            Err(e) => return Err(RldsError::Io(e)),
        }
        let data_len = u64::from_le_bytes(len_buf) as usize;

        // Skip CRC of length (u32)
        let mut crc_buf = [0u8; 4];
        reader.read_exact(&mut crc_buf)?;

        // Read data
        let mut data = vec![0u8; data_len];
        reader.read_exact(&mut data)?;

        // Skip CRC of data (u32)
        reader.read_exact(&mut crc_buf)?;

        // Parse the protobuf Example
        match parse_tf_example(&data) {
            Ok(step) => steps.push(step),
            Err(_) => {
                // Skip unparseable records silently
                continue;
            }
        }
    }

    Ok(steps)
}

// ── Protobuf parsing (manual, no prost codegen needed) ──────
//
// tf.train.Example is:
//   message Example { Features features = 1; }
//   message Features { map<string, Feature> feature = 1; }
//   message Feature { oneof kind { BytesList, FloatList, Int64List } }
//
// We parse this manually using protobuf wire format.

fn parse_tf_example(data: &[u8]) -> Result<RldsStep, RldsError> {
    let features = parse_features_from_example(data)?;

    let is_first = get_bool_feature(&features, "is_first").unwrap_or(false);
    let is_last = get_bool_feature(&features, "is_last").unwrap_or(false);
    let is_terminal = get_bool_feature(&features, "is_terminal").unwrap_or(false);
    let reward = get_float_feature(&features, "reward").unwrap_or(0.0);

    // Action: try "action" as float list
    let action = get_float_list_feature(&features, "action").unwrap_or_default();

    // State: concatenate all observation/* float fields (sorted by name)
    let mut state = Vec::new();
    let mut obs_keys: Vec<&String> = features
        .keys()
        .filter(|k| k.starts_with("observation/") && !k.contains("image"))
        .collect();
    obs_keys.sort();
    for key in &obs_keys {
        if let Some(FeatureValue::FloatList(vals)) = features.get(*key) {
            state.extend(vals);
        }
    }

    // Language instruction
    let language_instruction = get_bytes_feature(&features, "language_instruction")
        .or_else(|| get_bytes_feature(&features, "observation/language_instruction"))
        .and_then(|b| String::from_utf8(b).ok());

    // Images: observation/*image* fields (bytes that could be JPEG/PNG)
    let mut images = Vec::new();
    let mut img_keys: Vec<&String> = features.keys().filter(|k| k.contains("image")).collect();
    img_keys.sort();
    for key in &img_keys {
        if let Some(FeatureValue::BytesList(byte_entries)) = features.get(*key) {
            if let Some(img_bytes) = byte_entries.first() {
                // Try to decode as image to get dimensions
                if let Ok((rgb, w, h)) = decode_image_bytes(img_bytes) {
                    let camera = key
                        .replace("observation/", "")
                        .replace("image/", "")
                        .replace("image", "camera");
                    images.push((camera, rgb, w, h));
                }
            }
        }
    }

    Ok(RldsStep {
        is_first,
        is_last,
        is_terminal,
        action,
        state,
        reward,
        language_instruction,
        images,
    })
}

/// Decoded feature values.
#[derive(Debug)]
enum FeatureValue {
    FloatList(Vec<f32>),
    Int64List(Vec<i64>),
    BytesList(Vec<Vec<u8>>),
}

/// Parse protobuf Features from a tf.train.Example.
fn parse_features_from_example(data: &[u8]) -> Result<BTreeMap<String, FeatureValue>, RldsError> {
    let mut features = BTreeMap::new();
    let mut pos = 0;

    // Example { features: Features } — field 1, wire type 2 (length-delimited)
    // Features { map<string, Feature> } — each map entry is field 1, wire type 2

    // Skip outer Example wrapper
    let features_data = if !data.is_empty() {
        // Read field tag
        let (tag, new_pos) = read_varint(data, pos)?;
        pos = new_pos;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        if field_num == 1 && wire_type == 2 {
            let (len, new_pos) = read_varint(data, pos)?;
            pos = new_pos;
            &data[pos..pos + len as usize]
        } else {
            data
        }
    } else {
        return Ok(features);
    };

    // Parse Features map entries
    let mut fpos = 0;
    while fpos < features_data.len() {
        let (tag, new_pos) = read_varint(features_data, fpos)?;
        fpos = new_pos;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        if field_num == 1 && wire_type == 2 {
            // Map entry (length-delimited)
            let (entry_len, new_pos) = read_varint(features_data, fpos)?;
            fpos = new_pos;
            let entry_data = &features_data[fpos..fpos + entry_len as usize];
            fpos += entry_len as usize;

            // Parse map entry: key (field 1) and value (field 2)
            if let Ok((key, value)) = parse_map_entry(entry_data) {
                features.insert(key, value);
            }
        } else {
            // Skip unknown field
            fpos = skip_field(features_data, fpos, wire_type as u8)?;
        }
    }

    Ok(features)
}

fn parse_map_entry(data: &[u8]) -> Result<(String, FeatureValue), RldsError> {
    let mut pos = 0;
    let mut key = String::new();
    let mut value = None;

    while pos < data.len() {
        let (tag, new_pos) = read_varint(data, pos)?;
        pos = new_pos;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        if field_num == 1 && wire_type == 2 {
            // Key (string)
            let (len, new_pos) = read_varint(data, pos)?;
            pos = new_pos;
            key = String::from_utf8_lossy(&data[pos..pos + len as usize]).to_string();
            pos += len as usize;
        } else if field_num == 2 && wire_type == 2 {
            // Value (Feature message)
            let (len, new_pos) = read_varint(data, pos)?;
            pos = new_pos;
            let feature_data = &data[pos..pos + len as usize];
            pos += len as usize;
            value = parse_feature(feature_data).ok();
        } else {
            pos = skip_field(data, pos, wire_type as u8)?;
        }
    }

    match value {
        Some(v) => Ok((key, v)),
        None => Err(RldsError::Format("map entry without value".to_string())),
    }
}

fn parse_feature(data: &[u8]) -> Result<FeatureValue, RldsError> {
    if data.is_empty() {
        return Ok(FeatureValue::FloatList(vec![]));
    }

    let mut pos = 0;
    let (tag, new_pos) = read_varint(data, pos)?;
    pos = new_pos;
    let field_num = tag >> 3;
    let wire_type = tag & 0x7;

    if wire_type != 2 {
        return Err(RldsError::Format(
            "unexpected wire type in Feature".to_string(),
        ));
    }

    let (len, new_pos) = read_varint(data, pos)?;
    pos = new_pos;
    let inner = &data[pos..pos + len as usize];

    match field_num {
        1 => {
            // BytesList { value: repeated bytes }
            let mut entries = Vec::new();
            let mut ipos = 0;
            while ipos < inner.len() {
                let (tag, new_pos) = read_varint(inner, ipos)?;
                ipos = new_pos;
                let _fn = tag >> 3;
                let wt = tag & 0x7;
                if wt == 2 {
                    let (blen, new_pos) = read_varint(inner, ipos)?;
                    ipos = new_pos;
                    entries.push(inner[ipos..ipos + blen as usize].to_vec());
                    ipos += blen as usize;
                } else {
                    ipos = skip_field(inner, ipos, wt as u8)?;
                }
            }
            Ok(FeatureValue::BytesList(entries))
        }
        2 => {
            // FloatList { value: repeated float }
            let mut vals = Vec::new();
            if !inner.is_empty() {
                // Packed encoding: field 1, wire type 2
                let mut ipos = 0;
                let (tag, new_pos) = read_varint(inner, ipos)?;
                ipos = new_pos;
                let wt = tag & 0x7;
                if wt == 2 {
                    // Packed: length-delimited list of f32
                    let (plen, new_pos) = read_varint(inner, ipos)?;
                    ipos = new_pos;
                    let packed = &inner[ipos..ipos + plen as usize];
                    for chunk in packed.chunks_exact(4) {
                        vals.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                } else if wt == 5 {
                    // Non-packed: individual f32 values
                    vals.push(f32::from_le_bytes([
                        inner[ipos],
                        inner[ipos + 1],
                        inner[ipos + 2],
                        inner[ipos + 3],
                    ]));
                }
            }
            Ok(FeatureValue::FloatList(vals))
        }
        3 => {
            // Int64List { value: repeated int64 }
            let mut vals = Vec::new();
            if !inner.is_empty() {
                let mut ipos = 0;
                let (tag, new_pos) = read_varint(inner, ipos)?;
                ipos = new_pos;
                let wt = tag & 0x7;
                if wt == 2 {
                    let (plen, new_pos) = read_varint(inner, ipos)?;
                    ipos = new_pos;
                    let packed = &inner[ipos..ipos + plen as usize];
                    let mut ppos = 0;
                    while ppos < packed.len() {
                        let (val, new_pos) = read_varint_signed(packed, ppos)?;
                        ppos = new_pos;
                        vals.push(val);
                    }
                } else if wt == 0 {
                    let (val, _) = read_varint_signed(inner, ipos)?;
                    vals.push(val);
                }
            }
            Ok(FeatureValue::Int64List(vals))
        }
        _ => Err(RldsError::Format(format!(
            "unknown Feature field {}",
            field_num
        ))),
    }
}

// ── Protobuf varint helpers ─────────────────────────────────

fn read_varint(data: &[u8], mut pos: usize) -> Result<(u64, usize), RldsError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            return Err(RldsError::Format("truncated varint".to_string()));
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return Err(RldsError::Format("varint too long".to_string()));
        }
    }
}

fn read_varint_signed(data: &[u8], pos: usize) -> Result<(i64, usize), RldsError> {
    let (val, new_pos) = read_varint(data, pos)?;
    Ok((val as i64, new_pos))
}

fn skip_field(data: &[u8], pos: usize, wire_type: u8) -> Result<usize, RldsError> {
    match wire_type {
        0 => {
            // Varint
            let (_, new_pos) = read_varint(data, pos)?;
            Ok(new_pos)
        }
        1 => Ok(pos + 8), // 64-bit
        2 => {
            // Length-delimited
            let (len, new_pos) = read_varint(data, pos)?;
            Ok(new_pos + len as usize)
        }
        5 => Ok(pos + 4), // 32-bit
        _ => Err(RldsError::Format(format!(
            "unknown wire type {}",
            wire_type
        ))),
    }
}

// ── Feature extraction helpers ──────────────────────────────

fn get_bool_feature(features: &BTreeMap<String, FeatureValue>, key: &str) -> Option<bool> {
    match features.get(key) {
        Some(FeatureValue::Int64List(vals)) => vals.first().map(|&v| v != 0),
        _ => None,
    }
}

fn get_float_feature(features: &BTreeMap<String, FeatureValue>, key: &str) -> Option<f32> {
    match features.get(key) {
        Some(FeatureValue::FloatList(vals)) => vals.first().copied(),
        _ => None,
    }
}

fn get_float_list_feature(
    features: &BTreeMap<String, FeatureValue>,
    key: &str,
) -> Option<Vec<f32>> {
    match features.get(key) {
        Some(FeatureValue::FloatList(vals)) => Some(vals.clone()),
        _ => None,
    }
}

fn get_bytes_feature(features: &BTreeMap<String, FeatureValue>, key: &str) -> Option<Vec<u8>> {
    match features.get(key) {
        Some(FeatureValue::BytesList(entries)) => entries.first().cloned(),
        _ => None,
    }
}

// ── Image decoding ──────────────────────────────────────────

fn decode_image_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32), RldsError> {
    // Try to detect format from magic bytes and decode
    use image::ImageReader;
    use std::io::Cursor;

    let reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| RldsError::Format(format!("image format detection: {}", e)))?;

    let img = reader
        .decode()
        .map_err(|e| RldsError::Format(format!("image decode: {}", e)))?;

    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((rgb.into_raw(), w, h))
}

// ── Episode grouping ────────────────────────────────────────

fn group_into_episodes(steps: &[RldsStep]) -> Vec<Vec<&RldsStep>> {
    let mut episodes = Vec::new();
    let mut current: Vec<&RldsStep> = Vec::new();

    for step in steps {
        if step.is_first && !current.is_empty() {
            episodes.push(current);
            current = Vec::new();
        }
        current.push(step);
        if step.is_last || step.is_terminal {
            episodes.push(current);
            current = Vec::new();
        }
    }

    // Remaining steps form an incomplete episode
    if !current.is_empty() {
        episodes.push(current);
    }

    episodes
}

// ── File discovery ──────────────────────────────────────────

fn find_tfrecord_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_tfrecord_files(&path));
            } else if path.to_string_lossy().contains("tfrecord") {
                files.push(path);
            }
        }
    }
    files
}

fn read_dataset_task(dataset_dir: &Path) -> Option<String> {
    // Try to read from dataset_info.json
    let info_path = dataset_dir.join("dataset_info.json");
    if info_path.exists() {
        if let Ok(content) = fs::read_to_string(&info_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(desc) = val.get("description").and_then(|v| v.as_str()) {
                    return Some(desc.to_string());
                }
            }
        }
    }
    None
}
