//! Dataset mixtures — sample across multiple `.kdb` files with weights.
//!
//! A [`Mixture`] opens multiple `.kdb` files, each with a sampling weight,
//! and provides a unified interface for iterating episodes. This replaces
//! the manual `mixtures.py` that every VLA training codebase maintains.
//!
//! ```ignore
//! let mix = Mixture::builder()
//!     .add("bridge.kdb", 0.4)?
//!     .add("aloha.kdb", 0.3)?
//!     .add("libero.kdb", 0.3)?
//!     .seed(42)
//!     .build()?;
//!
//! // Iterate: each call picks a source according to weights
//! for _ in 0..1000 {
//!     let episode = mix.sample()?;
//! }
//!
//! // Or get a deterministic shuffled order
//! let order = mix.shuffled_indices();
//! ```

use crate::{Episode, EpisodeMeta, KdbReader, ReadError};

/// One source in a mixture.
struct MixSource {
    reader: KdbReader,
    path: String,
    /// Precomputed: which global indices belong to this source.
    /// global_start..global_start + reader.num_episodes()
    global_start: usize,
}

/// A weighted mixture of multiple `.kdb` datasets.
pub struct Mixture {
    sources: Vec<MixSource>,
    /// Total episodes across all sources.
    total_episodes: usize,
    /// Total frames across all sources.
    total_frames: u64,
    /// Normalized weights (sum to 1.0).
    weights: Vec<f64>,
    /// RNG state for sampling (simple xorshift64).
    rng_state: u64,
}

/// Builder for constructing a Mixture.
pub struct MixtureBuilder {
    entries: Vec<(String, f64)>,
    seed: u64,
}

/// Errors from mixture operations.
#[derive(Debug)]
pub enum MixError {
    Read(ReadError),
    NoSources,
    ZeroWeight { path: String },
    Io(std::io::Error),
}

impl std::fmt::Display for MixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixError::Read(e) => write!(f, "read error: {}", e),
            MixError::NoSources => write!(f, "mixture has no sources"),
            MixError::ZeroWeight { path } => write!(f, "zero weight for {}", path),
            MixError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for MixError {}

impl From<ReadError> for MixError {
    fn from(e: ReadError) -> Self {
        MixError::Read(e)
    }
}

impl From<std::io::Error> for MixError {
    fn from(e: std::io::Error) -> Self {
        MixError::Io(e)
    }
}

impl MixtureBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            seed: 42,
        }
    }

    /// Add a `.kdb` file with a weight (relative, will be normalized).
    pub fn add(mut self, path: impl Into<String>, weight: f64) -> Self {
        self.entries.push((path.into(), weight));
        self
    }

    /// Set the random seed for shuffling.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Build the mixture. Opens all files and normalizes weights.
    pub fn build(self) -> Result<Mixture, MixError> {
        if self.entries.is_empty() {
            return Err(MixError::NoSources);
        }

        let mut sources = Vec::with_capacity(self.entries.len());
        let mut total_episodes: usize = 0;
        let mut total_frames: u64 = 0;
        let mut raw_weights = Vec::with_capacity(self.entries.len());

        for (path, weight) in &self.entries {
            if *weight <= 0.0 {
                return Err(MixError::ZeroWeight { path: path.clone() });
            }

            let reader = KdbReader::open(path)?;
            let n = reader.num_episodes();
            let f = reader.num_frames();

            sources.push(MixSource {
                reader,
                path: path.clone(),
                global_start: total_episodes,
            });

            total_episodes += n;
            total_frames += f;
            raw_weights.push(*weight);
        }

        // Normalize weights
        let sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / sum).collect();

        Ok(Mixture {
            sources,
            total_episodes,
            total_frames,
            weights,
            rng_state: self.seed,
        })
    }
}

impl Default for MixtureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Mixture {
    /// Create a builder.
    pub fn builder() -> MixtureBuilder {
        MixtureBuilder::new()
    }

    /// Total episodes across all sources.
    pub fn total_episodes(&self) -> usize {
        self.total_episodes
    }

    /// Total frames across all sources.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Number of sources in this mixture.
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }

    /// Get info about each source: (path, num_episodes, weight).
    pub fn source_info(&self) -> Vec<(&str, usize, f64)> {
        self.sources
            .iter()
            .zip(self.weights.iter())
            .map(|(s, w)| (s.path.as_str(), s.reader.num_episodes(), *w))
            .collect()
    }

    /// Sample one episode according to the mixture weights.
    pub fn sample(&mut self) -> Result<Episode, MixError> {
        let source_idx = self.weighted_sample();
        let n = self.sources[source_idx].reader.num_episodes();
        let ep_idx = (self.xorshift64() as usize) % n;
        Ok(self.sources[source_idx].reader.read_episode(ep_idx)?)
    }

    /// Sample just the metadata (cheaper — no frame data loaded).
    pub fn sample_meta(&mut self) -> Result<EpisodeMeta, MixError> {
        let source_idx = self.weighted_sample();
        let n = self.sources[source_idx].reader.num_episodes();
        let ep_idx = (self.xorshift64() as usize) % n;
        Ok(self.sources[source_idx].reader.read_meta(ep_idx)?)
    }

    /// Read an episode by global index (0..total_episodes).
    /// Global index 0..N₁ maps to source 0, N₁..N₁+N₂ maps to source 1, etc.
    pub fn read_global(&self, global_idx: usize) -> Result<Episode, MixError> {
        let (source, local_idx) = self.resolve_global(global_idx)?;
        Ok(source.reader.read_episode(local_idx)?)
    }

    /// Generate a shuffled order of global indices, respecting weights.
    ///
    /// Returns `n` indices where each source appears proportional to its
    /// weight. Useful for creating a deterministic training epoch.
    pub fn weighted_epoch(&mut self, n: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(n);

        for _ in 0..n {
            let source_idx = self.weighted_sample();
            let n_ep = self.sources[source_idx].reader.num_episodes();
            let global_start = self.sources[source_idx].global_start;
            let ep_idx = (self.xorshift64() as usize) % n_ep;
            let global_idx = global_start + ep_idx;
            indices.push(global_idx);
        }

        indices
    }

    // ── Private helpers ─────────────────────────────────────

    /// Pick a source index according to normalized weights.
    fn weighted_sample(&mut self) -> usize {
        let r = (self.xorshift64() as f64) / (u64::MAX as f64);
        let mut cumulative = 0.0;
        for (i, &w) in self.weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                return i;
            }
        }
        self.weights.len() - 1
    }

    /// Resolve a global index to (source, local_index).
    fn resolve_global(&self, global_idx: usize) -> Result<(&MixSource, usize), MixError> {
        for source in self.sources.iter().rev() {
            if global_idx >= source.global_start {
                let local = global_idx - source.global_start;
                if local < source.reader.num_episodes() {
                    return Ok((source, local));
                }
            }
        }
        Err(MixError::Read(ReadError::EpisodeNotFound {
            position: global_idx,
        }))
    }

    /// Simple xorshift64 PRNG — fast, deterministic, no dependencies.
    fn xorshift64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Episode, EpisodeId, EpisodeMeta, Frame, KdbWriter};

    fn make_episode(embodiment: &str, task: &str, num_frames: u32) -> Episode {
        let meta = EpisodeMeta::new(EpisodeId(0), embodiment, task, num_frames, 7, 10.0);
        let frames = (0..num_frames)
            .map(|t| Frame {
                timestep: t,
                images: vec![],
                state: vec![0.0; 6],
                action: vec![0.0; 7],
                reward: Some(0.0),
                is_terminal: t == num_frames - 1,
            })
            .collect();
        Episode { meta, frames }
    }

    fn create_test_kdb(path: &str, embodiment: &str, task: &str, n: usize) {
        let mut writer = KdbWriter::create(path).unwrap();
        for _ in 0..n {
            writer
                .write_episode(&make_episode(embodiment, task, 10))
                .unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn mixture_basic() {
        let path_a = "/tmp/kinodb_mix_a.kdb";
        let path_b = "/tmp/kinodb_mix_b.kdb";
        create_test_kdb(path_a, "franka", "pick block", 5);
        create_test_kdb(path_b, "widowx", "open drawer", 3);

        let mix = Mixture::builder()
            .add(path_a, 0.6)
            .add(path_b, 0.4)
            .seed(123)
            .build()
            .unwrap();

        assert_eq!(mix.total_episodes(), 8);
        assert_eq!(mix.num_sources(), 2);

        let info = mix.source_info();
        assert_eq!(info[0].1, 5); // franka: 5 episodes
        assert_eq!(info[1].1, 3); // widowx: 3 episodes
        assert!((info[0].2 - 0.6).abs() < 1e-6);
        assert!((info[1].2 - 0.4).abs() < 1e-6);

        std::fs::remove_file(path_a).ok();
        std::fs::remove_file(path_b).ok();
    }

    #[test]
    fn mixture_sampling_distribution() {
        let path_a = "/tmp/kinodb_mix_dist_a.kdb";
        let path_b = "/tmp/kinodb_mix_dist_b.kdb";
        create_test_kdb(path_a, "franka", "task_a", 10);
        create_test_kdb(path_b, "widowx", "task_b", 10);

        let mut mix = Mixture::builder()
            .add(path_a, 0.8)
            .add(path_b, 0.2)
            .seed(42)
            .build()
            .unwrap();

        let mut count_a = 0u64;
        let n = 1000;

        for _ in 0..n {
            let meta = mix.sample_meta().unwrap();
            if meta.embodiment == "franka" {
                count_a += 1;
            }
        }

        // With 0.8/0.2 weights and 1000 samples, franka should be ~800
        let ratio = count_a as f64 / n as f64;
        assert!(
            ratio > 0.7 && ratio < 0.9,
            "Expected ~0.8 franka ratio, got {}",
            ratio
        );

        std::fs::remove_file(path_a).ok();
        std::fs::remove_file(path_b).ok();
    }

    #[test]
    fn mixture_read_global() {
        let path_a = "/tmp/kinodb_mix_global_a.kdb";
        let path_b = "/tmp/kinodb_mix_global_b.kdb";
        create_test_kdb(path_a, "franka", "task_a", 3);
        create_test_kdb(path_b, "widowx", "task_b", 2);

        let mix = Mixture::builder()
            .add(path_a, 0.5)
            .add(path_b, 0.5)
            .build()
            .unwrap();

        // Global 0,1,2 → source A; global 3,4 → source B
        let ep0 = mix.read_global(0).unwrap();
        assert_eq!(ep0.meta.embodiment, "franka");

        let ep3 = mix.read_global(3).unwrap();
        assert_eq!(ep3.meta.embodiment, "widowx");

        // Out of bounds
        assert!(mix.read_global(5).is_err());

        std::fs::remove_file(path_a).ok();
        std::fs::remove_file(path_b).ok();
    }

    #[test]
    fn mixture_weighted_epoch() {
        let path_a = "/tmp/kinodb_mix_epoch_a.kdb";
        let path_b = "/tmp/kinodb_mix_epoch_b.kdb";
        create_test_kdb(path_a, "franka", "a", 5);
        create_test_kdb(path_b, "widowx", "b", 5);

        let mut mix = Mixture::builder()
            .add(path_a, 0.5)
            .add(path_b, 0.5)
            .seed(99)
            .build()
            .unwrap();

        let epoch = mix.weighted_epoch(100);
        assert_eq!(epoch.len(), 100);

        // All indices should be valid (0..10)
        for &idx in &epoch {
            assert!(idx < 10);
        }

        // Both sources should appear
        let from_a = epoch.iter().filter(|&&i| i < 5).count();
        let from_b = epoch.iter().filter(|&&i| i >= 5).count();
        assert!(from_a > 20, "Expected more from source A, got {}", from_a);
        assert!(from_b > 20, "Expected more from source B, got {}", from_b);

        std::fs::remove_file(path_a).ok();
        std::fs::remove_file(path_b).ok();
    }

    #[test]
    fn mixture_no_sources() {
        let result = Mixture::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn mixture_deterministic_with_seed() {
        let path = "/tmp/kinodb_mix_det.kdb";
        create_test_kdb(path, "franka", "task", 10);

        let mut mix1 = Mixture::builder().add(path, 1.0).seed(42).build().unwrap();

        let mut mix2 = Mixture::builder().add(path, 1.0).seed(42).build().unwrap();

        // Same seed → same sequence
        for _ in 0..20 {
            let a = mix1.sample_meta().unwrap();
            let b = mix2.sample_meta().unwrap();
            assert_eq!(a.id, b.id);
        }

        std::fs::remove_file(path).ok();
    }
}
