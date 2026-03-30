//! Core data types for representing robot trajectory data.
//!
//! A trajectory dataset is made of [`Episode`]s. Each episode contains
//! a sequence of [`Frame`]s captured at regular timesteps. Each frame
//! has observations (images, proprioception), an action the robot took,
//! and optional metadata (language instruction, success, reward).
//!
//! These types are plain data — no file I/O, no dependencies.

/// A unique identifier for an episode within a database.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EpisodeId(pub u64);

/// Metadata about one complete episode (trajectory).
///
/// This is the "header" — it doesn't contain the actual frames,
/// just enough info to filter/query episodes without loading them.
#[derive(Debug, Clone)]
pub struct EpisodeMeta {
    /// Unique id within this database.
    pub id: EpisodeId,

    /// Which robot collected this episode (e.g. "widowx", "franka", "aloha").
    pub embodiment: String,

    /// Task description in natural language (e.g. "pick up the red block").
    pub language_instruction: String,

    /// Number of frames (timesteps) in this episode.
    pub num_frames: u32,

    /// Control frequency in Hz (e.g. 10.0 for LeRobot, 3.0 for Bridge).
    pub fps: f32,

    /// Dimensionality of the action vector (e.g. 7 for 6-DoF + gripper).
    pub action_dim: u16,

    /// Whether the episode ended in task success.
    pub success: Option<bool>,

    /// Total undiscounted reward (if available, e.g. from RL datasets).
    pub total_reward: Option<f32>,
}

/// One timestep of robot data.
///
/// A frame is the atomic unit — one "row" of the trajectory.
/// At each timestep the robot observes, acts, and gets a reward.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Index within the episode (0-based).
    pub timestep: u32,

    /// Camera observations as raw image bytes.
    /// Key = camera name (e.g. "front", "wrist"), Value = image data.
    /// For now we store raw bytes; later this becomes a reference into
    /// a video segment in the .kdb file.
    pub images: Vec<ImageObs>,

    /// Robot state / proprioception (e.g. joint positions, gripper state).
    /// Empty if not available for this dataset.
    pub state: Vec<f32>,

    /// The action the robot executed at this timestep.
    pub action: Vec<f32>,

    /// Per-step reward (if available).
    pub reward: Option<f32>,

    /// Whether this is the terminal frame of the episode.
    pub is_terminal: bool,
}

/// A single camera image observation.
#[derive(Debug, Clone)]
pub struct ImageObs {
    /// Camera name (e.g. "front", "wrist", "left_shoulder").
    pub camera: String,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,

    /// Number of channels (typically 3 for RGB).
    pub channels: u8,

    /// Raw pixel data (RGB, row-major, u8).
    /// Length = width * height * channels.
    pub data: Vec<u8>,
}

/// A complete episode: metadata + all its frames.
///
/// This is what you get when you fully load one episode from the database.
/// For large-scale training you'd stream frames instead of loading the
/// whole episode into memory — but this type is useful for inspection,
/// testing, and small datasets.
#[derive(Debug, Clone)]
pub struct Episode {
    pub meta: EpisodeMeta,
    pub frames: Vec<Frame>,
}

// Construction helpers

impl EpisodeMeta {
    /// Create a minimal episode metadata with required fields only.
    pub fn new(
        id: EpisodeId,
        embodiment: impl Into<String>,
        language_instruction: impl Into<String>,
        num_frames: u32,
        action_dim: u16,
        fps: f32,
    ) -> Self {
        Self {
            id,
            embodiment: embodiment.into(),
            language_instruction: language_instruction.into(),
            num_frames,
            fps,
            action_dim,
            success: None,
            total_reward: None,
        }
    }
}

impl ImageObs {
    /// Create a new image observation. Validates that data length matches dimensions.
    pub fn new(
        camera: impl Into<String>,
        width: u32,
        height: u32,
        channels: u8,
        data: Vec<u8>,
    ) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize) * (channels as usize);
        if data.len() != expected {
            return Err(format!(
                "image data length {} doesn't match {}x{}x{} = {}",
                data.len(),
                width,
                height,
                channels,
                expected
            ));
        }
        Ok(Self {
            camera: camera.into(),
            width,
            height,
            channels,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_id_equality() {
        let a = EpisodeId(42);
        let b = EpisodeId(42);
        assert_eq!(a, b);
    }

    #[test]
    fn episode_id_inequality() {
        let a = EpisodeId(1);
        let b = EpisodeId(2);
        assert_ne!(a, b);
    }

    #[test]
    fn create_episode_meta() {
        let meta = EpisodeMeta::new(
            EpisodeId(0),
            "widowx",
            "pick up the red block",
            100,
            7,
            10.0,
        );
        assert_eq!(meta.embodiment, "widowx");
        assert_eq!(meta.num_frames, 100);
        assert_eq!(meta.action_dim, 7);
        assert_eq!(meta.success, None);
    }

    #[test]
    fn create_image_obs_valid() {
        // 2x2 RGB image = 12 bytes
        let data = vec![0u8; 12];
        let img = ImageObs::new("front", 2, 2, 3, data);
        assert!(img.is_ok());
        assert_eq!(img.unwrap().camera, "front");
    }

    #[test]
    fn create_image_obs_wrong_size() {
        // 2x2 RGB image expects 12 bytes, we give 10
        let data = vec![0u8; 10];
        let img = ImageObs::new("front", 2, 2, 3, data);
        assert!(img.is_err());
    }

    #[test]
    fn create_frame() {
        let frame = Frame {
            timestep: 0,
            images: vec![],
            state: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            action: vec![0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
            reward: Some(0.0),
            is_terminal: false,
        };
        assert_eq!(frame.action.len(), 7);
        assert!(!frame.is_terminal);
    }

    #[test]
    fn build_complete_episode() {
        let meta = EpisodeMeta::new(EpisodeId(0), "franka", "open the drawer", 2, 7, 5.0);

        let frames = vec![
            Frame {
                timestep: 0,
                images: vec![],
                state: vec![0.0; 6],
                action: vec![0.0; 7],
                reward: Some(0.0),
                is_terminal: false,
            },
            Frame {
                timestep: 1,
                images: vec![],
                state: vec![0.1; 6],
                action: vec![0.1; 7],
                reward: Some(1.0),
                is_terminal: true,
            },
        ];

        let episode = Episode { meta, frames };

        assert_eq!(episode.meta.num_frames, 2);
        assert_eq!(episode.frames.len(), 2);
        assert!(episode.frames[1].is_terminal);
    }
}
