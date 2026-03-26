//! The episode index.
//!
//! The episode index is an array of fixed-size entries stored at the end
//! of the `.kdb` file (at the offset specified by the file header's
//! `index_offset` field). Each entry describes one episode: where its
//! action data and image data live in the file.
//!
//! ## Why a separate index?
//!
//! When a training loop asks for "episode #372, frames 10..20", we need
//! to jump directly to that data without scanning the whole file. The
//! index gives us the byte offsets to do exactly that.
//!
//! ## Binary layout per entry (64 bytes, little-endian)
//!
//! | Offset | Size | Field              | Description                                |
//! |--------|------|--------------------|--------------------------------------------|
//! | 0      | 8    | episode_id         | Unique episode identifier                  |
//! | 8      | 4    | num_frames         | Number of frames in this episode           |
//! | 12     | 2    | action_dim         | Dimensionality of the action vector        |
//! | 14     | 2    | state_dim          | Dimensionality of the state vector         |
//! | 16     | 8    | actions_offset     | Byte offset to this episode's action data  |
//! | 24     | 8    | actions_length     | Byte length of this episode's action data  |
//! | 32     | 8    | images_offset      | Byte offset to this episode's image data   |
//! | 40     | 8    | images_length      | Byte length of this episode's image data   |
//! | 48     | 8    | meta_offset        | Byte offset to this episode's metadata blob|
//! | 56     | 8    | meta_length        | Byte length of the metadata blob           |

use crate::EpisodeId;

/// Size of one index entry in bytes.
pub const INDEX_ENTRY_SIZE: usize = 64;

/// One entry in the episode index.
///
/// This is a fixed-size record. The full index is just `num_episodes`
/// of these packed back-to-back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry {
    /// Which episode this entry describes.
    pub episode_id: EpisodeId,
    /// Number of frames (timesteps) in this episode.
    pub num_frames: u32,
    /// Dimensionality of the action vector.
    pub action_dim: u16,
    /// Dimensionality of the state/proprioception vector.
    pub state_dim: u16,

    /// Byte offset in the .kdb file where this episode's actions start.
    pub actions_offset: u64,
    /// Byte length of this episode's action data.
    pub actions_length: u64,

    /// Byte offset in the .kdb file where this episode's images start.
    pub images_offset: u64,
    /// Byte length of this episode's image data.
    pub images_length: u64,

    /// Byte offset to a metadata blob (embodiment, task string, etc).
    pub meta_offset: u64,
    /// Byte length of the metadata blob.
    pub meta_length: u64,
}

impl IndexEntry {
    /// Serialize this entry to exactly 64 bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];

        buf[0..8].copy_from_slice(&self.episode_id.0.to_le_bytes());
        buf[8..12].copy_from_slice(&self.num_frames.to_le_bytes());
        buf[12..14].copy_from_slice(&self.action_dim.to_le_bytes());
        buf[14..16].copy_from_slice(&self.state_dim.to_le_bytes());
        buf[16..24].copy_from_slice(&self.actions_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.actions_length.to_le_bytes());
        buf[32..40].copy_from_slice(&self.images_offset.to_le_bytes());
        buf[40..48].copy_from_slice(&self.images_length.to_le_bytes());
        buf[48..56].copy_from_slice(&self.meta_offset.to_le_bytes());
        buf[56..64].copy_from_slice(&self.meta_length.to_le_bytes());

        buf
    }

    /// Deserialize one entry from a byte slice (must be >= 64 bytes).
    pub fn from_bytes(data: &[u8]) -> Result<Self, IndexError> {
        if data.len() < INDEX_ENTRY_SIZE {
            return Err(IndexError::EntryTooShort { got: data.len() });
        }

        Ok(Self {
            episode_id: EpisodeId(u64::from_le_bytes(data[0..8].try_into().unwrap())),
            num_frames: u32::from_le_bytes(data[8..12].try_into().unwrap()),
            action_dim: u16::from_le_bytes(data[12..14].try_into().unwrap()),
            state_dim: u16::from_le_bytes(data[14..16].try_into().unwrap()),
            actions_offset: u64::from_le_bytes(data[16..24].try_into().unwrap()),
            actions_length: u64::from_le_bytes(data[24..32].try_into().unwrap()),
            images_offset: u64::from_le_bytes(data[32..40].try_into().unwrap()),
            images_length: u64::from_le_bytes(data[40..48].try_into().unwrap()),
            meta_offset: u64::from_le_bytes(data[48..56].try_into().unwrap()),
            meta_length: u64::from_le_bytes(data[56..64].try_into().unwrap()),
        })
    }
}

/// The full episode index: a list of entries with lookup methods.
#[derive(Debug, Clone)]
pub struct EpisodeIndex {
    entries: Vec<IndexEntry>,
}

impl EpisodeIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create an index from a vec of entries.
    pub fn from_entries(entries: Vec<IndexEntry>) -> Self {
        Self { entries }
    }

    /// Add an entry to the index.
    pub fn push(&mut self, entry: IndexEntry) {
        self.entries.push(entry);
    }

    /// Number of episodes in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the index empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by its position (0-based).
    pub fn get(&self, pos: usize) -> Option<&IndexEntry> {
        self.entries.get(pos)
    }

    /// Find an entry by episode id. Linear scan for now — fine for
    /// thousands of episodes. We can add a HashMap later if needed.
    pub fn find(&self, id: EpisodeId) -> Option<&IndexEntry> {
        self.entries.iter().find(|e| e.episode_id == id)
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &IndexEntry> {
        self.entries.iter()
    }

    /// Serialize the entire index to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.entries.len() * INDEX_ENTRY_SIZE);
        for entry in &self.entries {
            buf.extend_from_slice(&entry.to_bytes());
        }
        buf
    }

    /// Deserialize an index from bytes.
    ///
    /// `data` must be an exact multiple of `INDEX_ENTRY_SIZE` (64 bytes).
    pub fn from_bytes(data: &[u8]) -> Result<Self, IndexError> {
        if data.len() % INDEX_ENTRY_SIZE != 0 {
            return Err(IndexError::BadAlignment {
                total_len: data.len(),
                entry_size: INDEX_ENTRY_SIZE,
            });
        }

        let num_entries = data.len() / INDEX_ENTRY_SIZE;
        let mut entries = Vec::with_capacity(num_entries);

        for i in 0..num_entries {
            let start = i * INDEX_ENTRY_SIZE;
            let end = start + INDEX_ENTRY_SIZE;
            entries.push(IndexEntry::from_bytes(&data[start..end])?);
        }

        Ok(Self { entries })
    }

    /// Total byte size of the serialized index.
    pub fn byte_size(&self) -> usize {
        self.entries.len() * INDEX_ENTRY_SIZE
    }
}

impl Default for EpisodeIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur when reading an index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexError {
    /// A single entry is shorter than 64 bytes.
    EntryTooShort { got: usize },
    /// The index data is not a multiple of 64 bytes.
    BadAlignment { total_len: usize, entry_size: usize },
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::EntryTooShort { got } => {
                write!(f, "index entry too short: expected {} bytes, got {}", INDEX_ENTRY_SIZE, got)
            }
            IndexError::BadAlignment { total_len, entry_size } => {
                write!(
                    f,
                    "index data length {} is not a multiple of entry size {}",
                    total_len, entry_size
                )
            }
        }
    }
}

impl std::error::Error for IndexError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(id: u64) -> IndexEntry {
        IndexEntry {
            episode_id: EpisodeId(id),
            num_frames: 100,
            action_dim: 7,
            state_dim: 6,
            actions_offset: 1000 + id * 5000,
            actions_length: 2800,
            images_offset: 50000 + id * 100000,
            images_length: 95000,
            meta_offset: 500 + id * 200,
            meta_length: 180,
        }
    }

    #[test]
    fn entry_roundtrip() {
        let entry = sample_entry(42);
        let bytes = entry.to_bytes();
        assert_eq!(bytes.len(), INDEX_ENTRY_SIZE);

        let parsed = IndexEntry::from_bytes(&bytes).unwrap();
        assert_eq!(entry, parsed);
    }

    #[test]
    fn entry_too_short() {
        let bytes = [0u8; 10];
        let err = IndexEntry::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, IndexError::EntryTooShort { got: 10 }));
    }

    #[test]
    fn index_roundtrip_multiple_entries() {
        let mut index = EpisodeIndex::new();
        for i in 0..5 {
            index.push(sample_entry(i));
        }
        assert_eq!(index.len(), 5);

        let bytes = index.to_bytes();
        assert_eq!(bytes.len(), 5 * INDEX_ENTRY_SIZE);

        let parsed = EpisodeIndex::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.len(), 5);

        // Verify each entry survived the roundtrip
        for i in 0..5 {
            assert_eq!(index.get(i), parsed.get(i));
        }
    }

    #[test]
    fn index_empty_roundtrip() {
        let index = EpisodeIndex::new();
        assert!(index.is_empty());

        let bytes = index.to_bytes();
        assert_eq!(bytes.len(), 0);

        let parsed = EpisodeIndex::from_bytes(&bytes).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn index_bad_alignment() {
        let bytes = [0u8; 100]; // not a multiple of 64
        let err = EpisodeIndex::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, IndexError::BadAlignment { .. }));
    }

    #[test]
    fn find_by_episode_id() {
        let mut index = EpisodeIndex::new();
        for i in 0..10 {
            index.push(sample_entry(i));
        }

        let found = index.find(EpisodeId(7));
        assert!(found.is_some());
        assert_eq!(found.unwrap().episode_id, EpisodeId(7));
        assert_eq!(found.unwrap().num_frames, 100);

        let not_found = index.find(EpisodeId(999));
        assert!(not_found.is_none());
    }

    #[test]
    fn byte_size_matches() {
        let mut index = EpisodeIndex::new();
        for i in 0..3 {
            index.push(sample_entry(i));
        }
        assert_eq!(index.byte_size(), 3 * INDEX_ENTRY_SIZE);
        assert_eq!(index.to_bytes().len(), index.byte_size());
    }
}
