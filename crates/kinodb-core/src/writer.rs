//! Write episodes to a `.kdb` file.
//!
//! ## File layout
//!
//! ```text
//! ┌──────────────────────────┐  offset 0
//! │  FileHeader (64 bytes)   │
//! ├──────────────────────────┤  offset 64
//! │  Episode 0: meta blob    │
//! │  Episode 0: actions      │
//! │  Episode 0: images       │
//! ├──────────────────────────┤
//! │  Episode 1: meta blob    │
//! │  Episode 1: actions      │
//! │  Episode 1: images       │
//! ├──────────────────────────┤
//! │  ...                     │
//! ├──────────────────────────┤  ← header.index_offset
//! │  EpisodeIndex            │
//! └──────────────────────────┘
//! ```
//!
//! The header is written last (we seek back to offset 0) because we
//! don't know the final counts and index offset until all episodes
//! have been written.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Episode, EpisodeId, EpisodeIndex, FileHeader, IndexEntry, HEADER_SIZE};

/// Writes episodes into a `.kdb` file.
///
/// Usage:
/// ```ignore
/// let mut writer = KdbWriter::create("output.kdb")?;
/// writer.write_episode(&episode1)?;
/// writer.write_episode(&episode2)?;
/// writer.finish()?;  // MUST call this to write the index + header
/// ```
pub struct KdbWriter {
    writer: BufWriter<File>,
    index: EpisodeIndex,
    total_frames: u64,
    next_id: u64,
    /// Current byte position in the file (tracked manually so we
    /// don't need to call seek just to know where we are).
    pos: u64,
}

/// Errors from the writer.
#[derive(Debug)]
pub enum WriteError {
    Io(io::Error),
    EmptyEpisode,
    InconsistentActionDim {
        episode_id: u64,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Io(e) => write!(f, "I/O error: {}", e),
            WriteError::EmptyEpisode => write!(f, "cannot write an episode with 0 frames"),
            WriteError::InconsistentActionDim {
                episode_id,
                expected,
                got,
            } => write!(
                f,
                "episode {}: expected action_dim={}, but frame has action.len()={}",
                episode_id, expected, got
            ),
        }
    }
}

impl std::error::Error for WriteError {}

impl From<io::Error> for WriteError {
    fn from(e: io::Error) -> Self {
        WriteError::Io(e)
    }
}

impl KdbWriter {
    /// Create a new `.kdb` file at the given path.
    ///
    /// Writes a placeholder header (64 zero bytes) that gets overwritten
    /// when you call [`finish()`](KdbWriter::finish).
    pub fn create(path: impl AsRef<Path>) -> Result<Self, WriteError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write placeholder header — will be overwritten in finish()
        let placeholder = [0u8; HEADER_SIZE];
        writer.write_all(&placeholder)?;

        Ok(Self {
            writer,
            index: EpisodeIndex::new(),
            total_frames: 0,
            next_id: 0,
            pos: HEADER_SIZE as u64,
        })
    }

    /// Write one episode to the file. Returns the assigned `EpisodeId`.
    pub fn write_episode(&mut self, episode: &Episode) -> Result<EpisodeId, WriteError> {
        if episode.frames.is_empty() {
            return Err(WriteError::EmptyEpisode);
        }

        let episode_id = EpisodeId(self.next_id);
        self.next_id += 1;

        // ── 1. Write metadata blob ──────────────────────────
        let meta_blob = self.encode_meta(&episode.meta);
        let meta_offset = self.pos;
        self.write_bytes(&meta_blob)?;
        let meta_length = meta_blob.len() as u64;

        // ── 2. Write actions (all frames, packed f32s) ──────
        let action_dim = episode.meta.action_dim as usize;
        let actions_offset = self.pos;
        for (_i, frame) in episode.frames.iter().enumerate() {
            if frame.action.len() != action_dim {
                return Err(WriteError::InconsistentActionDim {
                    episode_id: episode_id.0,
                    expected: action_dim,
                    got: frame.action.len(),
                });
            }
            // Write state first, then action for each frame
            self.write_f32_slice(&frame.state)?;
            self.write_f32_slice(&frame.action)?;

            // Write reward + is_terminal as a small footer per frame
            let reward = frame.reward.unwrap_or(0.0);
            self.write_f32(reward)?;
            self.write_u8(if frame.is_terminal { 1 } else { 0 })?;
        }
        let actions_length = self.pos - actions_offset;

        // Infer state_dim from the first frame
        let state_dim = episode.frames[0].state.len() as u16;

        // ── 3. Write images (all frames, all cameras) ───────
        let images_offset = self.pos;
        for frame in &episode.frames {
            // Write number of cameras for this frame (u16)
            self.write_u16(frame.images.len() as u16)?;
            for img in &frame.images {
                // Camera name: length-prefixed string
                self.write_string(&img.camera)?;
                // Dimensions
                self.write_u32(img.width)?;
                self.write_u32(img.height)?;
                self.write_u8(img.channels)?;
                // Raw pixel data: length-prefixed
                self.write_u32(img.data.len() as u32)?;
                self.write_bytes(&img.data)?;
            }
        }
        let images_length = self.pos - images_offset;

        // ── 4. Record in index ──────────────────────────────
        self.index.push(IndexEntry {
            episode_id,
            num_frames: episode.frames.len() as u32,
            action_dim: action_dim as u16,
            state_dim,
            actions_offset,
            actions_length,
            images_offset,
            images_length,
            meta_offset,
            meta_length,
        });

        self.total_frames += episode.frames.len() as u64;
        Ok(episode_id)
    }

    /// Finalize the file: write the index, then seek back and write the
    /// real header. You **must** call this or the file will be invalid.
    pub fn finish(mut self) -> Result<(), WriteError> {
        // Write the index
        let index_offset = self.pos;
        let index_bytes = self.index.to_bytes();
        self.write_bytes(&index_bytes)?;
        let index_length = index_bytes.len() as u64;

        // Build the real header
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let header = FileHeader {
            version_major: crate::VERSION_MAJOR,
            version_minor: crate::VERSION_MINOR,
            num_episodes: self.index.len() as u64,
            num_frames: self.total_frames,
            index_offset,
            index_length,
            created_timestamp: now,
        };

        // Seek back to the start and overwrite the placeholder
        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&header.to_bytes())?;
        self.writer.flush()?;

        Ok(())
    }

    // ── Private write helpers ───────────────────────────────

    fn write_bytes(&mut self, data: &[u8]) -> Result<(), WriteError> {
        self.writer.write_all(data)?;
        self.pos += data.len() as u64;
        Ok(())
    }

    fn write_f32_slice(&mut self, values: &[f32]) -> Result<(), WriteError> {
        for &v in values {
            self.write_f32(v)?;
        }
        Ok(())
    }

    fn write_f32(&mut self, value: f32) -> Result<(), WriteError> {
        let bytes = value.to_le_bytes();
        self.write_bytes(&bytes)
    }

    fn write_u8(&mut self, value: u8) -> Result<(), WriteError> {
        self.write_bytes(&[value])
    }

    fn write_u16(&mut self, value: u16) -> Result<(), WriteError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_u32(&mut self, value: u32) -> Result<(), WriteError> {
        self.write_bytes(&value.to_le_bytes())
    }

    fn write_string(&mut self, s: &str) -> Result<(), WriteError> {
        let bytes = s.as_bytes();
        // Length prefix (u16), then the string bytes
        self.write_u16(bytes.len() as u16)?;
        self.write_bytes(bytes)
    }

    /// Encode episode metadata as a simple blob.
    ///
    /// Format: length-prefixed strings + fixed fields.
    /// This is deliberately simple — we can switch to something
    /// more structured later without changing the index format
    /// (the index just stores offset + length).
    fn encode_meta(&self, meta: &crate::EpisodeMeta) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);

        // embodiment (length-prefixed string)
        let emb = meta.embodiment.as_bytes();
        buf.extend_from_slice(&(emb.len() as u16).to_le_bytes());
        buf.extend_from_slice(emb);

        // language_instruction (length-prefixed string)
        let lang = meta.language_instruction.as_bytes();
        buf.extend_from_slice(&(lang.len() as u16).to_le_bytes());
        buf.extend_from_slice(lang);

        // fps (f32)
        buf.extend_from_slice(&meta.fps.to_le_bytes());

        // success: 0 = unknown, 1 = false, 2 = true
        let success_byte: u8 = match meta.success {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        };
        buf.push(success_byte);

        // total_reward: present flag (u8) + f32 if present
        match meta.total_reward {
            None => buf.push(0),
            Some(r) => {
                buf.push(1);
                buf.extend_from_slice(&r.to_le_bytes());
            }
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Episode, EpisodeId, EpisodeMeta, Frame, ImageObs};
    use std::fs;

    /// Helper: create a small test episode.
    fn make_test_episode(num_frames: u32, with_images: bool) -> Episode {
        let meta = EpisodeMeta {
            id: EpisodeId(0), // writer assigns its own id
            embodiment: "widowx".to_string(),
            language_instruction: "pick up the red block".to_string(),
            num_frames,
            fps: 10.0,
            action_dim: 7,
            success: Some(true),
            total_reward: Some(1.0),
        };

        let frames = (0..num_frames)
            .map(|t| {
                let images = if with_images {
                    // tiny 2x2 RGB image
                    vec![ImageObs {
                        camera: "front".to_string(),
                        width: 2,
                        height: 2,
                        channels: 3,
                        data: vec![128u8; 12],
                    }]
                } else {
                    vec![]
                };

                Frame {
                    timestep: t,
                    images,
                    state: vec![0.1; 6],
                    action: vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                    reward: Some(if t == num_frames - 1 { 1.0 } else { 0.0 }),
                    is_terminal: t == num_frames - 1,
                }
            })
            .collect();

        Episode { meta, frames }
    }

    #[test]
    fn write_single_episode_no_images() {
        let path = "/tmp/kinodb_test_single.kdb";
        let episode = make_test_episode(5, false);

        let mut writer = KdbWriter::create(path).unwrap();
        let id = writer.write_episode(&episode).unwrap();
        assert_eq!(id, EpisodeId(0));
        writer.finish().unwrap();

        // File should exist and start with "KINO"
        let data = fs::read(path).unwrap();
        assert!(data.len() > HEADER_SIZE);
        assert_eq!(&data[0..4], b"KINO");

        // Parse the header
        let header = FileHeader::from_bytes(&data).unwrap();
        assert_eq!(header.num_episodes, 1);
        assert_eq!(header.num_frames, 5);
        assert!(header.index_offset > 0);

        fs::remove_file(path).ok();
    }

    #[test]
    fn write_multiple_episodes() {
        let path = "/tmp/kinodb_test_multi.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        for _ in 0..3 {
            let episode = make_test_episode(10, false);
            writer.write_episode(&episode).unwrap();
        }
        writer.finish().unwrap();

        let data = fs::read(path).unwrap();
        let header = FileHeader::from_bytes(&data).unwrap();
        assert_eq!(header.num_episodes, 3);
        assert_eq!(header.num_frames, 30);

        // Read back the index
        let idx_start = header.index_offset as usize;
        let idx_end = idx_start + header.index_length as usize;
        let index = EpisodeIndex::from_bytes(&data[idx_start..idx_end]).unwrap();
        assert_eq!(index.len(), 3);

        // Episode ids should be sequential
        assert_eq!(index.get(0).unwrap().episode_id, EpisodeId(0));
        assert_eq!(index.get(1).unwrap().episode_id, EpisodeId(1));
        assert_eq!(index.get(2).unwrap().episode_id, EpisodeId(2));

        fs::remove_file(path).ok();
    }

    #[test]
    fn write_with_images() {
        let path = "/tmp/kinodb_test_images.kdb";
        let episode = make_test_episode(3, true);

        let mut writer = KdbWriter::create(path).unwrap();
        writer.write_episode(&episode).unwrap();
        writer.finish().unwrap();

        let data = fs::read(path).unwrap();
        let header = FileHeader::from_bytes(&data).unwrap();
        assert_eq!(header.num_episodes, 1);

        // images_length should be > 0 in the index
        let idx_start = header.index_offset as usize;
        let idx_end = idx_start + header.index_length as usize;
        let index = EpisodeIndex::from_bytes(&data[idx_start..idx_end]).unwrap();
        assert!(index.get(0).unwrap().images_length > 0);

        fs::remove_file(path).ok();
    }

    #[test]
    fn reject_empty_episode() {
        let path = "/tmp/kinodb_test_empty.kdb";
        let episode = Episode {
            meta: EpisodeMeta::new(EpisodeId(0), "test", "test", 0, 7, 10.0),
            frames: vec![],
        };

        let mut writer = KdbWriter::create(path).unwrap();
        let result = writer.write_episode(&episode);
        assert!(result.is_err());

        fs::remove_file(path).ok();
    }

    #[test]
    fn reject_inconsistent_action_dim() {
        let path = "/tmp/kinodb_test_bad_action.kdb";
        let mut episode = make_test_episode(3, false);
        // Corrupt one frame's action to have wrong dimension
        episode.frames[1].action = vec![0.0; 5]; // should be 7

        let mut writer = KdbWriter::create(path).unwrap();
        let result = writer.write_episode(&episode);
        assert!(result.is_err());

        fs::remove_file(path).ok();
    }

    #[test]
    fn sequential_episode_ids() {
        let path = "/tmp/kinodb_test_seq_ids.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        let id0 = writer.write_episode(&make_test_episode(2, false)).unwrap();
        let id1 = writer.write_episode(&make_test_episode(2, false)).unwrap();
        let id2 = writer.write_episode(&make_test_episode(2, false)).unwrap();
        writer.finish().unwrap();

        assert_eq!(id0, EpisodeId(0));
        assert_eq!(id1, EpisodeId(1));
        assert_eq!(id2, EpisodeId(2));

        fs::remove_file(path).ok();
    }
}
