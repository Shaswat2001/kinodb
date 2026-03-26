//! Integration test: HDF5 → .kdb roundtrip.
//!
//! Creates a fake robomimic-style HDF5 file using the hdf5 crate,
//! ingests it into a .kdb file, reads it back, and verifies the data.

use hdf5;
use kinodb_core::{EpisodeId, KdbReader};
use kinodb_ingest::hdf5::{ingest_hdf5, Hdf5IngestConfig};
use ndarray::{Array1, Array2, Array4};

/// Create a minimal robomimic-style HDF5 file with the given parameters.
fn create_test_hdf5(
    path: &str,
    num_demos: usize,
    frames_per_demo: usize,
    action_dim: usize,
    with_images: bool,
) {
    let file = hdf5::File::create(path).expect("failed to create HDF5 file");
    let data = file
        .create_group("data")
        .expect("failed to create data group");

    for demo_idx in 0..num_demos {
        let demo_name = format!("demo_{}", demo_idx);
        let demo = data
            .create_group(&demo_name)
            .expect("failed to create demo group");

        // actions: (N, action_dim) float32
        let actions = Array2::<f32>::zeros((frames_per_demo, action_dim));
        demo.new_dataset_builder()
            .with_data(&actions)
            .create("actions")
            .expect("failed to write actions");

        // rewards: (N,) float32
        let mut rewards = Array1::<f32>::zeros(frames_per_demo);
        rewards[frames_per_demo - 1] = 1.0; // success
        demo.new_dataset_builder()
            .with_data(&rewards)
            .create("rewards")
            .expect("failed to write rewards");

        // dones: (N,) float32
        let mut dones = Array1::<f32>::zeros(frames_per_demo);
        dones[frames_per_demo - 1] = 1.0;
        demo.new_dataset_builder()
            .with_data(&dones)
            .create("dones")
            .expect("failed to write dones");

        // obs group
        let obs = demo
            .create_group("obs")
            .expect("failed to create obs group");

        // robot0_eef_pos: (N, 3) float32 — a state key
        let eef_pos = Array2::<f32>::from_shape_fn((frames_per_demo, 3), |(t, d)| {
            (t as f32) * 0.01 + (d as f32) * 0.1
        });
        obs.new_dataset_builder()
            .with_data(&eef_pos)
            .create("robot0_eef_pos")
            .expect("failed to write eef_pos");

        // robot0_gripper_qpos: (N, 2) float32 — another state key
        let gripper = Array2::<f32>::ones((frames_per_demo, 2));
        obs.new_dataset_builder()
            .with_data(&gripper)
            .create("robot0_gripper_qpos")
            .expect("failed to write gripper");

        if with_images {
            // agentview_image: (N, 8, 8, 3) uint8 — tiny test images
            let img = Array4::<u8>::from_shape_fn((frames_per_demo, 8, 8, 3), |(t, y, x, c)| {
                ((t + y + x + c) % 256) as u8
            });
            obs.new_dataset_builder()
                .with_data(&img)
                .create("agentview_image")
                .expect("failed to write image");
        }
    }
}

#[test]
fn hdf5_roundtrip_no_images() {
    let hdf5_path = "/tmp/kinodb_test_ingest_noimg.hdf5";
    let kdb_path = "/tmp/kinodb_test_ingest_noimg.kdb";

    create_test_hdf5(hdf5_path, 3, 20, 7, false);

    let config = Hdf5IngestConfig {
        embodiment: "franka".to_string(),
        task: Some("pick up the block".to_string()),
        fps: 20.0,
        max_episodes: None,
    };

    let result = ingest_hdf5(hdf5_path, kdb_path, &config).unwrap();
    assert_eq!(result.num_episodes, 3);
    assert_eq!(result.total_frames, 60);

    // Read back and verify
    let reader = KdbReader::open(kdb_path).unwrap();
    assert_eq!(reader.num_episodes(), 3);
    assert_eq!(reader.num_frames(), 60);

    let ep = reader.read_episode(0).unwrap();
    assert_eq!(ep.meta.embodiment, "franka");
    assert_eq!(ep.meta.language_instruction, "pick up the block");
    assert_eq!(ep.meta.fps, 20.0);
    assert_eq!(ep.meta.action_dim, 7);
    assert_eq!(ep.meta.num_frames, 20);
    assert_eq!(ep.frames.len(), 20);

    // State should be eef_pos (3) + gripper (2) = 5
    assert_eq!(ep.frames[0].state.len(), 5);

    // Actions should be 7-dim
    assert_eq!(ep.frames[0].action.len(), 7);

    // No images
    assert!(ep.frames[0].images.is_empty());

    // Last frame should be terminal with reward
    assert!(ep.frames[19].is_terminal);
    assert_eq!(ep.frames[19].reward, Some(1.0));

    // Success should be detected
    assert_eq!(ep.meta.success, Some(true));

    std::fs::remove_file(hdf5_path).ok();
    std::fs::remove_file(kdb_path).ok();
}

#[test]
fn hdf5_roundtrip_with_images() {
    let hdf5_path = "/tmp/kinodb_test_ingest_img.hdf5";
    let kdb_path = "/tmp/kinodb_test_ingest_img.kdb";

    create_test_hdf5(hdf5_path, 2, 10, 4, true);

    let config = Hdf5IngestConfig {
        embodiment: "widowx".to_string(),
        task: Some("open drawer".to_string()),
        fps: 10.0,
        max_episodes: None,
    };

    let result = ingest_hdf5(hdf5_path, kdb_path, &config).unwrap();
    assert_eq!(result.num_episodes, 2);
    assert_eq!(result.total_frames, 20);

    let reader = KdbReader::open(kdb_path).unwrap();
    let ep = reader.read_episode(0).unwrap();

    // Should have 1 camera
    assert_eq!(ep.frames[0].images.len(), 1);
    let img = &ep.frames[0].images[0];
    assert_eq!(img.camera, "agentview_image");
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
    assert_eq!(img.channels, 3);
    assert_eq!(img.data.len(), 8 * 8 * 3);

    // Verify image pixel data survived roundtrip
    // Pixel at (t=0, y=0, x=0, c=0) should be (0+0+0+0)%256 = 0
    assert_eq!(img.data[0], 0);
    // Pixel at (t=0, y=1, x=2, c=1) = (0+1+2+1)%256 = 4
    let pixel_idx = (1 * 8 * 3) + (2 * 3) + 1;
    assert_eq!(img.data[pixel_idx], 4);

    std::fs::remove_file(hdf5_path).ok();
    std::fs::remove_file(kdb_path).ok();
}

#[test]
fn hdf5_max_episodes_limit() {
    let hdf5_path = "/tmp/kinodb_test_ingest_limit.hdf5";
    let kdb_path = "/tmp/kinodb_test_ingest_limit.kdb";

    create_test_hdf5(hdf5_path, 10, 5, 3, false);

    let config = Hdf5IngestConfig {
        embodiment: "ur5".to_string(),
        task: None, // should fall back to filename
        fps: 5.0,
        max_episodes: Some(3),
    };

    let result = ingest_hdf5(hdf5_path, kdb_path, &config).unwrap();
    assert_eq!(result.num_episodes, 3);
    assert_eq!(result.total_frames, 15);

    let reader = KdbReader::open(kdb_path).unwrap();
    assert_eq!(reader.num_episodes(), 3);

    // Task should be the filename since we didn't provide one
    let meta = reader.read_meta(0).unwrap();
    assert_eq!(meta.language_instruction, "kinodb_test_ingest_limit");

    // Episode IDs should be 0, 1, 2
    assert_eq!(reader.read_episode(0).unwrap().meta.id, EpisodeId(0));
    assert_eq!(reader.read_episode(1).unwrap().meta.id, EpisodeId(1));
    assert_eq!(reader.read_episode(2).unwrap().meta.id, EpisodeId(2));

    std::fs::remove_file(hdf5_path).ok();
    std::fs::remove_file(kdb_path).ok();
}

#[test]
fn hdf5_missing_data_group() {
    let hdf5_path = "/tmp/kinodb_test_ingest_bad.hdf5";
    let kdb_path = "/tmp/kinodb_test_ingest_bad.kdb";

    // Create an empty HDF5 file (no data/ group)
    let _file = hdf5::File::create(hdf5_path).unwrap();

    let config = Hdf5IngestConfig::default();
    let result = ingest_hdf5(hdf5_path, kdb_path, &config);
    assert!(result.is_err());

    std::fs::remove_file(hdf5_path).ok();
    std::fs::remove_file(kdb_path).ok();
}
