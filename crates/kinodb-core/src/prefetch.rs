//! Background prefetch reader for kinodb.
//!
//! Wraps a `KdbReader` with a dedicated thread that pre-loads episodes
//! into a bounded channel. The training loop calls `next()` and gets
//! an already-decoded `Episode` — no I/O wait.
//!
//! ```ignore
//! use kinodb_core::prefetch::PrefetchReader;
//!
//! let prefetch = PrefetchReader::new("data.kdb", 16)?;  // buffer 16 episodes ahead
//!
//! // Sequential
//! for ep in prefetch.iter() {
//!     // ep is already decoded — zero wait
//! }
//!
//! // Or with a custom order
//! let prefetch = PrefetchReader::with_order("data.kdb", 16, vec![5, 2, 8, 0, 3])?;
//! ```

use std::sync::mpsc;
use std::thread;

use crate::{Episode, KdbReader, ReadError};

/// A prefetching reader that loads episodes in a background thread.
///
/// The background thread reads episodes in the specified order and
/// pushes them into a bounded channel. The main thread pulls from the
/// channel with zero I/O latency (as long as the prefetch keeps up).
pub struct PrefetchReader {
    receiver: mpsc::Receiver<Result<Episode, ReadError>>,
    _handle: thread::JoinHandle<()>,
    num_episodes: usize,
    num_frames: u64,
}

impl PrefetchReader {
    /// Create a prefetch reader that reads all episodes sequentially.
    ///
    /// `buffer_size` is how many episodes to read ahead (16–64 is typical).
    pub fn new(path: &str, buffer_size: usize) -> Result<Self, ReadError> {
        let reader = KdbReader::open(path)?;
        let n = reader.num_episodes();
        let order: Vec<usize> = (0..n).collect();
        Self::from_reader(reader, buffer_size, order)
    }

    /// Create a prefetch reader with a custom episode order.
    ///
    /// Useful for shuffled training epochs.
    pub fn with_order(
        path: &str,
        buffer_size: usize,
        order: Vec<usize>,
    ) -> Result<Self, ReadError> {
        let reader = KdbReader::open(path)?;
        Self::from_reader(reader, buffer_size, order)
    }

    fn from_reader(
        reader: KdbReader,
        buffer_size: usize,
        order: Vec<usize>,
    ) -> Result<Self, ReadError> {
        let num_episodes = reader.num_episodes();
        let num_frames = reader.num_frames();

        let buf = buffer_size.max(1);
        let (sender, receiver) = mpsc::sync_channel::<Result<Episode, ReadError>>(buf);

        let handle = thread::spawn(move || {
            for &pos in &order {
                let result = reader.read_episode(pos);
                // If the receiver is dropped (training stopped early), just exit
                if sender.send(result).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            receiver,
            _handle: handle,
            num_episodes,
            num_frames,
        })
    }

    /// Number of episodes in the underlying database.
    pub fn num_episodes(&self) -> usize {
        self.num_episodes
    }

    /// Total frames in the underlying database.
    pub fn num_frames(&self) -> u64 {
        self.num_frames
    }

    /// Get the next prefetched episode. Blocks until one is available.
    /// Returns `None` when all episodes in the order have been consumed.
    pub fn next(&self) -> Option<Result<Episode, ReadError>> {
        self.receiver.recv().ok()
    }

    /// Try to get the next episode without blocking.
    /// Returns `None` if no episode is ready yet or all have been consumed.
    pub fn try_next(&self) -> Option<Result<Episode, ReadError>> {
        self.receiver.try_recv().ok()
    }

    /// Iterate over all prefetched episodes.
    pub fn iter(&self) -> PrefetchIter<'_> {
        PrefetchIter { reader: self }
    }
}

/// Iterator over prefetched episodes.
pub struct PrefetchIter<'a> {
    reader: &'a PrefetchReader,
}

impl<'a> Iterator for PrefetchIter<'a> {
    type Item = Result<Episode, ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.reader.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeId, EpisodeMeta, Frame, KdbWriter};

    fn make_test_db(path: &str, n_episodes: usize) {
        let mut writer = KdbWriter::create(path).unwrap();
        for i in 0..n_episodes {
            let ep = crate::Episode {
                meta: EpisodeMeta {
                    id: EpisodeId(0),
                    embodiment: "test".to_string(),
                    language_instruction: format!("task_{}", i),
                    num_frames: 5,
                    fps: 10.0,
                    action_dim: 3,
                    success: Some(true),
                    total_reward: Some(1.0),
                },
                frames: (0..5)
                    .map(|t| Frame {
                        timestep: t as u32,
                        images: vec![],
                        state: vec![i as f32; 4],
                        action: vec![0.01; 3],
                        reward: Some(0.0),
                        is_terminal: t == 4,
                    })
                    .collect(),
            };
            writer.write_episode(&ep).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn prefetch_sequential() {
        let path = "/tmp/kinodb_prefetch_seq.kdb";
        make_test_db(path, 20);

        let prefetch = PrefetchReader::new(path, 4).unwrap();
        let mut count = 0;
        for result in prefetch.iter() {
            let ep = result.unwrap();
            assert_eq!(ep.frames.len(), 5);
            count += 1;
        }
        assert_eq!(count, 20);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn prefetch_custom_order() {
        let path = "/tmp/kinodb_prefetch_order.kdb";
        make_test_db(path, 10);

        let order = vec![9, 0, 5, 3, 7];
        let prefetch = PrefetchReader::with_order(path, 2, order.clone()).unwrap();

        let mut received_tasks = Vec::new();
        for result in prefetch.iter() {
            let ep = result.unwrap();
            received_tasks.push(ep.meta.language_instruction.clone());
        }

        assert_eq!(received_tasks.len(), 5);
        // Verify order matches
        for (i, &pos) in order.iter().enumerate() {
            assert_eq!(received_tasks[i], format!("task_{}", pos));
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn prefetch_early_drop() {
        let path = "/tmp/kinodb_prefetch_drop.kdb";
        make_test_db(path, 100);

        let prefetch = PrefetchReader::new(path, 8).unwrap();

        // Read only 5 episodes then drop
        for _ in 0..5 {
            let _ = prefetch.next().unwrap().unwrap();
        }
        drop(prefetch);

        // Background thread should exit cleanly (no panic)
        std::fs::remove_file(path).ok();
    }
}
