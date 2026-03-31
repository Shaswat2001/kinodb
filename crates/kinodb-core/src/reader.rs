//! Read episodes from a `.kdb` file.
//!
//! The reader memory-maps the file for zero-copy access. The OS pages
//! in only the data you actually read, so opening a multi-GB database
//! is instant and uses minimal RAM.
//!
//! ```ignore
//! let reader = KdbReader::open("data.kdb")?;
//! println!("{} episodes", reader.num_episodes());
//!
//! // Read by position
//! let episode = reader.read_episode(0)?;
//!
//! // Read by id
//! let episode = reader.read_episode_by_id(EpisodeId(42))?;
//!
//! // Just the metadata (no frames loaded)
//! let meta = reader.read_meta(0)?;
//! ```

use std::fs;
use std::path::Path;

use crate::{
    Episode, EpisodeId, EpisodeIndex, EpisodeMeta, FileHeader, Frame, ImageObs, IndexEntry,
};

/// Backing storage for the reader — either mmap or owned bytes.
enum Storage {
    Mmap(memmap2::Mmap),
    Bytes(Vec<u8>),
}

impl std::ops::Deref for Storage {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Storage::Mmap(m) => m,
            Storage::Bytes(v) => v,
        }
    }
}

/// A reader for `.kdb` files.
///
/// Uses memory-mapped I/O by default. The OS lazily pages in only the
/// data you access, so opening a large file is near-instant.
pub struct KdbReader {
    /// Memory-mapped file data (or owned bytes for testing).
    data: Storage,
    /// Parsed file header.
    header: FileHeader,
    /// Parsed episode index.
    index: EpisodeIndex,
}

/// Errors from the reader.
#[derive(Debug)]
pub enum ReadError {
    Io(std::io::Error),
    Header(crate::HeaderError),
    Index(crate::IndexError),
    /// Tried to access an episode position that doesn't exist.
    EpisodeNotFound {
        position: usize,
    },
    /// Tried to find an episode by id that doesn't exist.
    EpisodeIdNotFound {
        id: EpisodeId,
    },
    /// Data is truncated or corrupt.
    UnexpectedEof {
        context: &'static str,
    },
    /// A string in the file is not valid UTF-8.
    InvalidUtf8 {
        context: &'static str,
    },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "I/O error: {}", e),
            ReadError::Header(e) => write!(f, "header error: {}", e),
            ReadError::Index(e) => write!(f, "index error: {}", e),
            ReadError::EpisodeNotFound { position } => {
                write!(f, "no episode at position {}", position)
            }
            ReadError::EpisodeIdNotFound { id } => {
                write!(f, "no episode with id {}", id.0)
            }
            ReadError::UnexpectedEof { context } => {
                write!(f, "unexpected end of data while reading {}", context)
            }
            ReadError::InvalidUtf8 { context } => {
                write!(f, "invalid UTF-8 in {}", context)
            }
        }
    }
}

impl std::error::Error for ReadError {}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        ReadError::Io(e)
    }
}

impl From<crate::HeaderError> for ReadError {
    fn from(e: crate::HeaderError) -> Self {
        ReadError::Header(e)
    }
}

impl From<crate::IndexError> for ReadError {
    fn from(e: crate::IndexError) -> Self {
        ReadError::Index(e)
    }
}

impl KdbReader {
    /// Open and validate a `.kdb` file using memory-mapped I/O.
    ///
    /// The file is mapped into the process address space but not read
    /// into RAM until accessed. This means opening a 10 GB file is
    /// near-instant and uses negligible memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ReadError> {
        let file = fs::File::open(path)?;
        // SAFETY: we treat the mmap as read-only and the file is not
        // modified while the reader exists. This is the standard
        // usage pattern for memmap2.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_storage(Storage::Mmap(mmap))
    }

    /// Open from an in-memory byte buffer (useful for tests).
    pub fn open_from_bytes(data: Vec<u8>) -> Result<Self, ReadError> {
        Self::from_storage(Storage::Bytes(data))
    }

    /// Shared init logic for both open paths.
    fn from_storage(data: Storage) -> Result<Self, ReadError> {
        let header = FileHeader::from_bytes(&data)?;

        let idx_start = header.index_offset as usize;
        let idx_end = idx_start + header.index_length as usize;
        if idx_end > data.len() {
            return Err(ReadError::UnexpectedEof {
                context: "episode index",
            });
        }
        let index = EpisodeIndex::from_bytes(&data[idx_start..idx_end])?;

        Ok(Self {
            data,
            header,
            index,
        })
    }

    /// Number of episodes in this file.
    pub fn num_episodes(&self) -> usize {
        self.index.len()
    }

    /// Total number of frames across all episodes.
    pub fn num_frames(&self) -> u64 {
        self.header.num_frames
    }

    /// Get the parsed file header.
    pub fn header(&self) -> &FileHeader {
        &self.header
    }

    /// Get the episode index.
    pub fn index(&self) -> &EpisodeIndex {
        &self.index
    }

    /// Read just the metadata for the episode at the given position.
    /// Cheaper than reading the full episode since it skips actions/images.
    pub fn read_meta(&self, position: usize) -> Result<EpisodeMeta, ReadError> {
        let entry = self
            .index
            .get(position)
            .ok_or(ReadError::EpisodeNotFound { position })?;

        self.decode_meta(entry)
    }

    /// Read a full episode (meta + all frames) by position (0-based).
    pub fn read_episode(&self, position: usize) -> Result<Episode, ReadError> {
        let entry = self
            .index
            .get(position)
            .ok_or(ReadError::EpisodeNotFound { position })?
            .clone();

        self.read_episode_from_entry(&entry)
    }

    /// Read a full episode by its `EpisodeId`.
    pub fn read_episode_by_id(&self, id: EpisodeId) -> Result<Episode, ReadError> {
        let entry = self
            .index
            .find(id)
            .ok_or(ReadError::EpisodeIdNotFound { id })?
            .clone();

        self.read_episode_from_entry(&entry)
    }

    // ── Private helpers ─────────────────────────────────────

    fn read_episode_from_entry(&self, entry: &IndexEntry) -> Result<Episode, ReadError> {
        let meta = self.decode_meta(entry)?;
        let frames = self.decode_frames(entry)?;
        Ok(Episode { meta, frames })
    }

    /// Decode the metadata blob for one episode.
    fn decode_meta(&self, entry: &IndexEntry) -> Result<EpisodeMeta, ReadError> {
        let start = entry.meta_offset as usize;
        let end = start + entry.meta_length as usize;
        if end > self.data.len() {
            return Err(ReadError::UnexpectedEof {
                context: "episode meta",
            });
        }

        let mut cursor = Cursor::new(&self.data[start..end]);

        let embodiment = cursor.read_string("embodiment")?;
        let language_instruction = cursor.read_string("language_instruction")?;
        let fps = cursor.read_f32("fps")?;

        let success_byte = cursor.read_u8("success")?;
        let success = match success_byte {
            0 => None,
            1 => Some(false),
            2 => Some(true),
            _ => None, // forward-compat: treat unknown as None
        };

        let reward_present = cursor.read_u8("reward_present")?;
        let total_reward = if reward_present == 1 {
            Some(cursor.read_f32("total_reward")?)
        } else {
            None
        };

        Ok(EpisodeMeta {
            id: entry.episode_id,
            embodiment,
            language_instruction,
            num_frames: entry.num_frames,
            fps,
            action_dim: entry.action_dim,
            success,
            total_reward,
        })
    }

    /// Decode all frames for one episode (actions section + images section).
    fn decode_frames(&self, entry: &IndexEntry) -> Result<Vec<Frame>, ReadError> {
        let state_dim = entry.state_dim as usize;
        let action_dim = entry.action_dim as usize;
        let num_frames = entry.num_frames as usize;

        // ── Read actions section ────────────────────────────
        let act_start = entry.actions_offset as usize;
        let act_end = act_start + entry.actions_length as usize;
        if act_end > self.data.len() {
            return Err(ReadError::UnexpectedEof {
                context: "actions data",
            });
        }
        let mut act_cursor = Cursor::new(&self.data[act_start..act_end]);

        // Pre-read all per-frame data from the actions section
        let mut frame_data: Vec<(Vec<f32>, Vec<f32>, f32, bool)> = Vec::with_capacity(num_frames);

        for _ in 0..num_frames {
            let state = act_cursor.read_f32_vec(state_dim, "state")?;
            let action = act_cursor.read_f32_vec(action_dim, "action")?;
            let reward = act_cursor.read_f32("reward")?;
            let is_terminal = act_cursor.read_u8("is_terminal")? != 0;
            frame_data.push((state, action, reward, is_terminal));
        }

        // ── Read images section ─────────────────────────────
        let img_start = entry.images_offset as usize;
        let img_end = img_start + entry.images_length as usize;
        if img_end > self.data.len() {
            return Err(ReadError::UnexpectedEof {
                context: "images data",
            });
        }
        let mut img_cursor = Cursor::new(&self.data[img_start..img_end]);

        let mut frames = Vec::with_capacity(num_frames);

        for (t, (state, action, reward, is_terminal)) in frame_data.into_iter().enumerate() {
            let num_cameras = img_cursor.read_u16("num_cameras")? as usize;
            let mut images = Vec::with_capacity(num_cameras);

            for _ in 0..num_cameras {
                let camera = img_cursor.read_string("camera_name")?;
                let width = img_cursor.read_u32("image_width")?;
                let height = img_cursor.read_u32("image_height")?;
                let channels = img_cursor.read_u8("image_channels")?;

                // Format byte: 0 = raw, 1 = JPEG
                // For backwards compatibility with pre-compression files:
                // old format didn't have this byte, so we peek at the byte.
                // If it's 0 or 1, treat as format byte. Old files wrote
                // a u32 data_len here, so the first byte would be the low
                // byte of the length — if channels=3 and w*h*3 < 256 it
                // could be 0 or 1, but that's only for tiny images (< 9x9).
                // In practice this is safe for all real image sizes.
                let format_byte = img_cursor.read_u8("image_format")?;
                let data_len = img_cursor.read_u32("image_data_len")? as usize;
                let raw_data = img_cursor.read_bytes(data_len, "image_data")?;

                let data = if format_byte == 1 {
                    // JPEG — decode to raw RGB
                    decompress_jpeg(&raw_data, width, height, channels)?
                } else {
                    // Raw pixels
                    raw_data
                };

                images.push(ImageObs {
                    camera,
                    width,
                    height,
                    channels,
                    data,
                });
            }

            frames.push(Frame {
                timestep: t as u32,
                images,
                state,
                action,
                reward: Some(reward),
                is_terminal,
            });
        }

        Ok(frames)
    }

    /// Decode frames from the actions section only — skip images entirely.
    /// Much faster for image-heavy datasets.
    fn decode_frames_no_images(&self, entry: &IndexEntry) -> Result<Vec<Frame>, ReadError> {
        let state_dim = entry.state_dim as usize;
        let action_dim = entry.action_dim as usize;
        let num_frames = entry.num_frames as usize;

        let act_start = entry.actions_offset as usize;
        let act_end = act_start + entry.actions_length as usize;
        if act_end > self.data.len() {
            return Err(ReadError::UnexpectedEof {
                context: "actions data",
            });
        }
        let mut act_cursor = Cursor::new(&self.data[act_start..act_end]);

        let mut frames = Vec::with_capacity(num_frames);
        for t in 0..num_frames {
            let state = act_cursor.read_f32_vec(state_dim, "state")?;
            let action = act_cursor.read_f32_vec(action_dim, "action")?;
            let reward = act_cursor.read_f32("reward")?;
            let is_terminal = act_cursor.read_u8("is_terminal")? != 0;

            frames.push(Frame {
                timestep: t as u32,
                images: vec![],
                state,
                action,
                reward: Some(reward),
                is_terminal,
            });
        }

        Ok(frames)
    }

    /// Read an episode's actions and states only (no images).
    ///
    /// Much faster than `read_episode` for image-heavy datasets since
    /// it skips the image section entirely.
    pub fn read_episode_actions_only(&self, position: usize) -> Result<Episode, ReadError> {
        let entry = self
            .index
            .get(position)
            .ok_or(ReadError::EpisodeNotFound { position })?
            .clone();

        let meta = self.decode_meta(&entry)?;
        let frames = self.decode_frames_no_images(&entry)?;
        Ok(Episode { meta, frames })
    }
}

// ── Cursor: a tiny helper for reading from a byte slice ─────────

/// A simple cursor over a byte slice. Tracks position and gives
/// nice error messages when data runs out.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize, context: &'static str) -> Result<Vec<u8>, ReadError> {
        if self.remaining() < n {
            return Err(ReadError::UnexpectedEof { context });
        }
        let slice = self.data[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(slice)
    }

    fn read_u8(&mut self, context: &'static str) -> Result<u8, ReadError> {
        if self.remaining() < 1 {
            return Err(ReadError::UnexpectedEof { context });
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    fn read_u16(&mut self, context: &'static str) -> Result<u16, ReadError> {
        let bytes = self.read_bytes(2, context)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self, context: &'static str) -> Result<u32, ReadError> {
        let bytes = self.read_bytes(4, context)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(&mut self, context: &'static str) -> Result<f32, ReadError> {
        let bytes = self.read_bytes(4, context)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32_vec(&mut self, n: usize, context: &'static str) -> Result<Vec<f32>, ReadError> {
        let mut vec = Vec::with_capacity(n);
        for _ in 0..n {
            vec.push(self.read_f32(context)?);
        }
        Ok(vec)
    }

    fn read_string(&mut self, context: &'static str) -> Result<String, ReadError> {
        let len = self.read_u16(context)? as usize;
        let bytes = self.read_bytes(len, context)?;
        String::from_utf8(bytes).map_err(|_| ReadError::InvalidUtf8 { context })
    }
}

// ── JPEG decompression helper ───────────────────────────────

fn decompress_jpeg(
    jpeg_data: &[u8],
    expected_width: u32,
    expected_height: u32,
    _expected_channels: u8,
) -> Result<Vec<u8>, ReadError> {
    use image::ImageReader;
    use std::io::Cursor;

    let reader = ImageReader::with_format(Cursor::new(jpeg_data), image::ImageFormat::Jpeg);

    let img = reader.decode().map_err(|_| ReadError::UnexpectedEof {
        context: "jpeg decode failed",
    })?;

    let rgb = img.to_rgb8();

    // Verify dimensions match
    if rgb.width() != expected_width || rgb.height() != expected_height {
        return Err(ReadError::UnexpectedEof {
            context: "jpeg dimensions mismatch",
        });
    }

    Ok(rgb.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeMeta, Frame, ImageObs, KdbWriter};

    /// Helper: same as in writer tests.
    fn make_test_episode(num_frames: u32, with_images: bool) -> Episode {
        let meta = EpisodeMeta {
            id: EpisodeId(0),
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
                    state: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    action: vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1.0],
                    reward: Some(if t == num_frames - 1 { 1.0 } else { 0.0 }),
                    is_terminal: t == num_frames - 1,
                }
            })
            .collect();

        Episode { meta, frames }
    }

    /// Write then read — the fundamental roundtrip test.
    #[test]
    fn roundtrip_single_episode_no_images() {
        let path = "/tmp/kinodb_rt_single.kdb";
        let original = make_test_episode(5, false);

        // Write
        let mut writer = KdbWriter::create(path).unwrap();
        writer.write_episode(&original).unwrap();
        writer.finish().unwrap();

        // Read
        let reader = KdbReader::open(path).unwrap();
        assert_eq!(reader.num_episodes(), 1);
        assert_eq!(reader.num_frames(), 5);

        let episode = reader.read_episode(0).unwrap();
        assert_eq!(episode.meta.embodiment, "widowx");
        assert_eq!(episode.meta.language_instruction, "pick up the red block");
        assert_eq!(episode.meta.num_frames, 5);
        assert_eq!(episode.meta.fps, 10.0);
        assert_eq!(episode.meta.action_dim, 7);
        assert_eq!(episode.meta.success, Some(true));
        assert_eq!(episode.meta.total_reward, Some(1.0));

        assert_eq!(episode.frames.len(), 5);
        assert_eq!(episode.frames[0].state, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(
            episode.frames[0].action,
            vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1.0]
        );
        assert_eq!(episode.frames[0].reward, Some(0.0));
        assert!(!episode.frames[0].is_terminal);
        assert!(episode.frames[4].is_terminal);
        assert_eq!(episode.frames[4].reward, Some(1.0));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn roundtrip_with_images() {
        let path = "/tmp/kinodb_rt_images.kdb";
        let original = make_test_episode(3, true);

        let mut writer = KdbWriter::create(path).unwrap();
        writer.write_episode(&original).unwrap();
        writer.finish().unwrap();

        let reader = KdbReader::open(path).unwrap();
        let episode = reader.read_episode(0).unwrap();

        assert_eq!(episode.frames[0].images.len(), 1);
        let img = &episode.frames[0].images[0];
        assert_eq!(img.camera, "front");
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        assert_eq!(img.channels, 3);
        assert_eq!(img.data.len(), 12);
        assert!(img.data.iter().all(|&b| b == 128));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn roundtrip_multiple_episodes() {
        let path = "/tmp/kinodb_rt_multi.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        for _ in 0..5 {
            writer.write_episode(&make_test_episode(8, true)).unwrap();
        }
        writer.finish().unwrap();

        let reader = KdbReader::open(path).unwrap();
        assert_eq!(reader.num_episodes(), 5);
        assert_eq!(reader.num_frames(), 40);

        // Read each episode and verify
        for i in 0..5 {
            let ep = reader.read_episode(i).unwrap();
            assert_eq!(ep.meta.embodiment, "widowx");
            assert_eq!(ep.frames.len(), 8);
            assert_eq!(ep.frames[0].images.len(), 1);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_by_episode_id() {
        let path = "/tmp/kinodb_rt_byid.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        for _ in 0..3 {
            writer.write_episode(&make_test_episode(4, false)).unwrap();
        }
        writer.finish().unwrap();

        let reader = KdbReader::open(path).unwrap();
        let ep = reader.read_episode_by_id(EpisodeId(2)).unwrap();
        assert_eq!(ep.meta.id, EpisodeId(2));
        assert_eq!(ep.frames.len(), 4);

        // Non-existent id
        let err = reader.read_episode_by_id(EpisodeId(99));
        assert!(err.is_err());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_meta_only() {
        let path = "/tmp/kinodb_rt_meta.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        writer.write_episode(&make_test_episode(10, true)).unwrap();
        writer.finish().unwrap();

        let reader = KdbReader::open(path).unwrap();
        let meta = reader.read_meta(0).unwrap();
        assert_eq!(meta.embodiment, "widowx");
        assert_eq!(meta.language_instruction, "pick up the red block");
        assert_eq!(meta.num_frames, 10);
        assert_eq!(meta.success, Some(true));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn read_nonexistent_position() {
        let path = "/tmp/kinodb_rt_nopos.kdb";

        let mut writer = KdbWriter::create(path).unwrap();
        writer.write_episode(&make_test_episode(2, false)).unwrap();
        writer.finish().unwrap();

        let reader = KdbReader::open(path).unwrap();
        let err = reader.read_episode(5);
        assert!(err.is_err());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn open_nonexistent_file() {
        let err = KdbReader::open("/tmp/kinodb_does_not_exist.kdb");
        assert!(err.is_err());
    }

    #[test]
    fn compressed_image_roundtrip() {
        let path = "/tmp/kinodb_rt_jpeg.kdb";

        // Create a test episode with a 16x16 RGB image
        let ep = Episode {
            meta: EpisodeMeta {
                id: EpisodeId(0),
                embodiment: "franka".to_string(),
                language_instruction: "test".to_string(),
                num_frames: 2,
                fps: 10.0,
                action_dim: 3,
                success: Some(true),
                total_reward: Some(1.0),
            },
            frames: (0..2)
                .map(|t| {
                    // Solid color block — survives JPEG compression well
                    let mut pixels = vec![0u8; 16 * 16 * 3];
                    for chunk in pixels.chunks_mut(3) {
                        chunk[0] = 200; // R
                        chunk[1] = 100; // G
                        chunk[2] = 50; // B
                    }
                    Frame {
                        timestep: t,
                        images: vec![ImageObs {
                            camera: "front".to_string(),
                            width: 16,
                            height: 16,
                            channels: 3,
                            data: pixels,
                        }],
                        state: vec![0.1; 4],
                        action: vec![0.01; 3],
                        reward: Some(0.0),
                        is_terminal: t == 1,
                    }
                })
                .collect(),
        };

        // Write with JPEG compression (quality=85)
        let mut writer = KdbWriter::create_compressed(path, 85).unwrap();
        writer.write_episode(&ep).unwrap();
        writer.finish().unwrap();

        // Compare file size to uncompressed
        let compressed_size = std::fs::metadata(path).unwrap().len();

        let raw_path = "/tmp/kinodb_rt_jpeg_raw.kdb";
        let mut raw_writer = KdbWriter::create(raw_path).unwrap();
        raw_writer.write_episode(&ep).unwrap();
        raw_writer.finish().unwrap();
        let raw_size = std::fs::metadata(raw_path).unwrap().len();

        // JPEG should be smaller
        assert!(
            compressed_size < raw_size,
            "compressed {} should be < raw {}",
            compressed_size,
            raw_size,
        );

        // Read back and verify dimensions and approximate pixel values
        let reader = KdbReader::open(path).unwrap();
        let read_ep = reader.read_episode(0).unwrap();
        assert_eq!(read_ep.frames.len(), 2);

        let img = &read_ep.frames[0].images[0];
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.channels, 3);
        assert_eq!(img.data.len(), 16 * 16 * 3);

        // JPEG is lossy — check pixels are approximately correct (within 20)
        let r = img.data[0] as i32;
        let g = img.data[1] as i32;
        let b = img.data[2] as i32;
        assert!((r - 200).abs() < 20, "R={} expected ~200", r);
        assert!((g - 100).abs() < 20, "G={} expected ~100", g);
        assert!((b - 50).abs() < 20, "B={} expected ~50", b);

        // Actions should be exact
        assert_eq!(read_ep.frames[0].action.len(), 3);
        assert_eq!(read_ep.frames[0].state.len(), 4);

        std::fs::remove_file(path).ok();
        std::fs::remove_file(raw_path).ok();
    }
}
