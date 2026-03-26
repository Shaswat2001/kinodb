//! The `.kdb` file header.
//!
//! Every `.kdb` file starts with a 64-byte header. This is enough to
//! identify the file, check compatibility, and know how much data is
//! inside without reading anything else.
//!
//! ## Binary layout (64 bytes, little-endian)
//!
//! | Offset | Size | Field              | Description                          |
//! |--------|------|--------------------|--------------------------------------|
//! | 0      | 4    | magic              | `b"KINO"` — identifies a .kdb file   |
//! | 4      | 2    | version_major      | Format major version (currently 0)   |
//! | 6      | 2    | version_minor      | Format minor version (currently 1)   |
//! | 8      | 8    | num_episodes       | Total number of episodes             |
//! | 16     | 8    | num_frames         | Total number of frames across all episodes |
//! | 24     | 8    | index_offset       | Byte offset to the episode index section |
//! | 32     | 8    | index_length       | Byte length of the episode index section |
//! | 40     | 8    | created_timestamp  | Unix timestamp (seconds) when file was created |
//! | 48     | 16   | _reserved          | Reserved for future use (must be 0)  |

/// Magic bytes at the start of every `.kdb` file.
pub const MAGIC: [u8; 4] = *b"KINO";

/// Current format version.
pub const VERSION_MAJOR: u16 = 0;
pub const VERSION_MINOR: u16 = 1;

/// Total size of the header in bytes.
pub const HEADER_SIZE: usize = 64;

/// The 64-byte header at the start of every `.kdb` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileHeader {
    /// Format major version.
    pub version_major: u16,
    /// Format minor version.
    pub version_minor: u16,
    /// Total number of episodes in this file.
    pub num_episodes: u64,
    /// Total number of frames across all episodes.
    pub num_frames: u64,
    /// Byte offset where the episode index starts.
    pub index_offset: u64,
    /// Byte length of the episode index.
    pub index_length: u64,
    /// Unix timestamp (seconds) when this file was created.
    pub created_timestamp: u64,
}

/// Errors that can occur when reading a header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeaderError {
    /// Input is shorter than 64 bytes.
    TooShort { got: usize },
    /// First 4 bytes are not `b"KINO"`.
    BadMagic { got: [u8; 4] },
    /// Version is newer than what we support.
    UnsupportedVersion { major: u16, minor: u16 },
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeaderError::TooShort { got } => {
                write!(
                    f,
                    "header too short: expected {} bytes, got {}",
                    HEADER_SIZE, got
                )
            }
            HeaderError::BadMagic { got } => {
                write!(f, "bad magic: expected {:?}, got {:?}", MAGIC, got)
            }
            HeaderError::UnsupportedVersion { major, minor } => {
                write!(
                    f,
                    "unsupported version {}.{} (we support up to {}.{})",
                    major, minor, VERSION_MAJOR, VERSION_MINOR
                )
            }
        }
    }
}

impl std::error::Error for HeaderError {}

impl FileHeader {
    /// Create a new header with the current version.
    pub fn new(num_episodes: u64, num_frames: u64, created_timestamp: u64) -> Self {
        Self {
            version_major: VERSION_MAJOR,
            version_minor: VERSION_MINOR,
            num_episodes,
            num_frames,
            index_offset: 0,
            index_length: 0,
            created_timestamp,
        }
    }

    /// Serialize this header to exactly 64 bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        // offset 0: magic
        buf[0..4].copy_from_slice(&MAGIC);
        // offset 4: version_major
        buf[4..6].copy_from_slice(&self.version_major.to_le_bytes());
        // offset 6: version_minor
        buf[6..8].copy_from_slice(&self.version_minor.to_le_bytes());
        // offset 8: num_episodes
        buf[8..16].copy_from_slice(&self.num_episodes.to_le_bytes());
        // offset 16: num_frames
        buf[16..24].copy_from_slice(&self.num_frames.to_le_bytes());
        // offset 24: index_offset
        buf[24..32].copy_from_slice(&self.index_offset.to_le_bytes());
        // offset 32: index_length
        buf[32..40].copy_from_slice(&self.index_length.to_le_bytes());
        // offset 40: created_timestamp
        buf[40..48].copy_from_slice(&self.created_timestamp.to_le_bytes());
        // offset 48..64: reserved (already zeroed)

        buf
    }

    /// Deserialize a header from a byte slice. Validates magic and version.
    pub fn from_bytes(data: &[u8]) -> Result<Self, HeaderError> {
        if data.len() < HEADER_SIZE {
            return Err(HeaderError::TooShort { got: data.len() });
        }

        // Check magic
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&data[0..4]);
        if magic != MAGIC {
            return Err(HeaderError::BadMagic { got: magic });
        }

        let version_major = u16::from_le_bytes([data[4], data[5]]);
        let version_minor = u16::from_le_bytes([data[6], data[7]]);

        // We reject files from a newer major version.
        if version_major > VERSION_MAJOR {
            return Err(HeaderError::UnsupportedVersion {
                major: version_major,
                minor: version_minor,
            });
        }

        let num_episodes = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let num_frames = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let index_offset = u64::from_le_bytes(data[24..32].try_into().unwrap());
        let index_length = u64::from_le_bytes(data[32..40].try_into().unwrap());
        let created_timestamp = u64::from_le_bytes(data[40..48].try_into().unwrap());

        Ok(Self {
            version_major,
            version_minor,
            num_episodes,
            num_frames,
            index_offset,
            index_length,
            created_timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_header() {
        let header = FileHeader {
            version_major: 0,
            version_minor: 1,
            num_episodes: 500,
            num_frames: 25_000,
            index_offset: 64,
            index_length: 4096,
            created_timestamp: 1_711_000_000,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), 64);

        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, parsed);
    }

    #[test]
    fn magic_bytes_correct() {
        let header = FileHeader::new(0, 0, 0);
        let bytes = header.to_bytes();
        assert_eq!(&bytes[0..4], b"KINO");
    }

    #[test]
    fn reject_bad_magic() {
        let mut bytes = FileHeader::new(0, 0, 0).to_bytes();
        bytes[0] = b'X'; // corrupt magic
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, HeaderError::BadMagic { .. }));
    }

    #[test]
    fn reject_too_short() {
        let bytes = [0u8; 10];
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, HeaderError::TooShort { got: 10 }));
    }

    #[test]
    fn reject_future_major_version() {
        let mut header = FileHeader::new(0, 0, 0);
        header.version_major = 99;
        let bytes = header.to_bytes();

        // Re-inject valid magic since to_bytes uses MAGIC constant
        let err = FileHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(
            err,
            HeaderError::UnsupportedVersion { major: 99, .. }
        ));
    }

    #[test]
    fn accept_same_major_newer_minor() {
        let mut header = FileHeader::new(10, 500, 1_700_000_000);
        header.version_minor = 99; // newer minor is fine
        let bytes = header.to_bytes();

        let parsed = FileHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.version_minor, 99);
        assert_eq!(parsed.num_episodes, 10);
    }

    #[test]
    fn reserved_bytes_are_zero() {
        let header = FileHeader::new(0, 0, 0);
        let bytes = header.to_bytes();
        // bytes 48..64 should all be zero
        assert!(bytes[48..64].iter().all(|&b| b == 0));
    }

    #[test]
    fn header_display_errors() {
        // Just make sure Display doesn't panic
        let e1 = HeaderError::TooShort { got: 5 };
        let e2 = HeaderError::BadMagic { got: [0, 0, 0, 0] };
        let e3 = HeaderError::UnsupportedVersion { major: 2, minor: 0 };
        let _ = format!("{}", e1);
        let _ = format!("{}", e2);
        let _ = format!("{}", e3);
    }
}
