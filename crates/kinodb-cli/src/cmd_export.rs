use kinodb_core::KdbReader;
use std::fs;
use std::path::Path;

pub fn run(
    kdb_path: &str,
    output_dir: &str,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        "numpy" => run_numpy(kdb_path, output_dir),
        "json" => run_json(kdb_path, output_dir),
        other => {
            eprintln!(
                "Unsupported export format: '{}'. Supported: numpy, json",
                other
            );
            std::process::exit(1);
        }
    }
}

/// Export to a directory of numpy-compatible binary files.
///
/// Output structure:
/// ```text
/// output_dir/
///   meta.json              ← database-level metadata
///   episode_000/
///     meta.json            ← episode metadata (embodiment, task, etc.)
///     actions.bin          ← (num_frames, action_dim) float32 little-endian
///     states.bin           ← (num_frames, state_dim) float32 little-endian
///     rewards.bin          ← (num_frames,) float32 little-endian
///     front.bin            ← (num_frames, H, W, C) uint8 [per camera]
///     wrist.bin            ← (num_frames, H, W, C) uint8 [per camera]
///   episode_001/
///     ...
/// ```
///
/// Python loading:
/// ```python
/// import numpy as np, json
/// meta = json.load(open("episode_000/meta.json"))
/// actions = np.fromfile("episode_000/actions.bin", dtype=np.float32)
///           .reshape(meta["num_frames"], meta["action_dim"])
/// ```
fn run_numpy(kdb_path: &str, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let reader = KdbReader::open(kdb_path)?;
    let header = reader.header();

    println!(
        "Exporting {} episodes from {}",
        header.num_episodes, kdb_path
    );
    println!("  Format: numpy (binary f32/u8 + JSON metadata)");
    println!("  Output: {}/", output_dir);
    println!();

    fs::create_dir_all(output_dir)?;

    // Database-level metadata
    let db_meta = format!(
        "{{\n  \"kinodb_version\": \"{}.{}\",\n  \"num_episodes\": {},\n  \"num_frames\": {},\n  \"source\": \"{}\"\n}}\n",
        header.version_major, header.version_minor,
        header.num_episodes,
        header.num_frames,
        kdb_path,
    );
    fs::write(Path::new(output_dir).join("meta.json"), db_meta)?;

    for i in 0..reader.num_episodes() {
        let episode = reader.read_episode(i)?;
        let ep_dir = Path::new(output_dir).join(format!("episode_{:03}", i));
        fs::create_dir_all(&ep_dir)?;

        let meta = &episode.meta;
        let frames = &episode.frames;
        let num_frames = frames.len();
        let action_dim = meta.action_dim as usize;
        let state_dim = if num_frames > 0 {
            frames[0].state.len()
        } else {
            0
        };

        // ── meta.json ───────────────────────────────────────
        let cameras_json: Vec<String> = if num_frames > 0 && !frames[0].images.is_empty() {
            frames[0].images.iter().map(|img| {
                format!(
                    "    {{\n      \"name\": \"{}\",\n      \"width\": {},\n      \"height\": {},\n      \"channels\": {}\n    }}",
                    img.camera, img.width, img.height, img.channels
                )
            }).collect()
        } else {
            vec![]
        };

        let ep_meta = format!(
            concat!(
                "{{\n",
                "  \"episode_id\": {},\n",
                "  \"embodiment\": \"{}\",\n",
                "  \"task\": \"{}\",\n",
                "  \"num_frames\": {},\n",
                "  \"action_dim\": {},\n",
                "  \"state_dim\": {},\n",
                "  \"fps\": {},\n",
                "  \"success\": {},\n",
                "  \"total_reward\": {},\n",
                "  \"cameras\": [\n{}\n  ]\n",
                "}}\n",
            ),
            meta.id.0,
            meta.embodiment,
            meta.language_instruction,
            num_frames,
            action_dim,
            state_dim,
            meta.fps,
            match meta.success {
                Some(true) => "true",
                Some(false) => "false",
                None => "null",
            },
            match meta.total_reward {
                Some(r) => format!("{}", r),
                None => "null".to_string(),
            },
            cameras_json.join(",\n"),
        );
        fs::write(ep_dir.join("meta.json"), ep_meta)?;

        // ── actions.bin ─────────────────────────────────────
        let mut actions_buf: Vec<u8> = Vec::with_capacity(num_frames * action_dim * 4);
        for frame in frames {
            for &val in &frame.action {
                actions_buf.extend_from_slice(&val.to_le_bytes());
            }
        }
        fs::write(ep_dir.join("actions.bin"), &actions_buf)?;

        // ── states.bin ──────────────────────────────────────
        if state_dim > 0 {
            let mut states_buf: Vec<u8> = Vec::with_capacity(num_frames * state_dim * 4);
            for frame in frames {
                for &val in &frame.state {
                    states_buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            fs::write(ep_dir.join("states.bin"), &states_buf)?;
        }

        // ── rewards.bin ─────────────────────────────────────
        let mut rewards_buf: Vec<u8> = Vec::with_capacity(num_frames * 4);
        for frame in frames {
            let r = frame.reward.unwrap_or(0.0);
            rewards_buf.extend_from_slice(&r.to_le_bytes());
        }
        fs::write(ep_dir.join("rewards.bin"), &rewards_buf)?;

        // ── Camera images ───────────────────────────────────
        // One .bin file per camera, all frames concatenated
        if num_frames > 0 && !frames[0].images.is_empty() {
            let num_cameras = frames[0].images.len();
            for cam_idx in 0..num_cameras {
                let cam_name = &frames[0].images[cam_idx].camera;
                let mut img_buf: Vec<u8> = Vec::new();
                for frame in frames {
                    if cam_idx < frame.images.len() {
                        img_buf.extend_from_slice(&frame.images[cam_idx].data);
                    }
                }
                let filename = format!("{}.bin", sanitize_filename(cam_name));
                fs::write(ep_dir.join(&filename), &img_buf)?;
            }
        }

        if (i + 1) % 50 == 0 || i == reader.num_episodes() - 1 {
            println!("  exported {}/{} episodes", i + 1, reader.num_episodes());
        }
    }

    // ── Write a Python loader script ────────────────────────
    let loader_script = r#""""Load kinodb numpy export into Python.

Usage:
    from load_kinodb import load_episode, load_dataset

    # Single episode
    ep = load_episode("episode_000")
    print(ep["actions"].shape)  # (num_frames, action_dim)

    # All episodes
    dataset = load_dataset(".")
    for ep in dataset:
        print(ep["meta"]["task"], ep["actions"].shape)
"""
import json
import numpy as np
from pathlib import Path


def load_episode(episode_dir):
    """Load one exported episode directory."""
    episode_dir = Path(episode_dir)
    meta = json.loads((episode_dir / "meta.json").read_text())

    n = meta["num_frames"]
    ad = meta["action_dim"]
    sd = meta["state_dim"]

    result = {"meta": meta}

    result["actions"] = np.fromfile(
        episode_dir / "actions.bin", dtype=np.float32
    ).reshape(n, ad)

    if sd > 0 and (episode_dir / "states.bin").exists():
        result["states"] = np.fromfile(
            episode_dir / "states.bin", dtype=np.float32
        ).reshape(n, sd)

    result["rewards"] = np.fromfile(
        episode_dir / "rewards.bin", dtype=np.float32
    )

    # Load camera images
    for cam in meta.get("cameras", []):
        name = cam["name"]
        h, w, c = cam["height"], cam["width"], cam["channels"]
        bin_path = episode_dir / f"{name}.bin"
        if bin_path.exists():
            result[f"image_{name}"] = np.fromfile(
                bin_path, dtype=np.uint8
            ).reshape(n, h, w, c)

    return result


def load_dataset(export_dir):
    """Load all episodes from an export directory."""
    export_dir = Path(export_dir)
    episodes = []
    for ep_dir in sorted(export_dir.glob("episode_*")):
        if ep_dir.is_dir():
            episodes.append(load_episode(ep_dir))
    return episodes


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    dataset = load_dataset(path)
    print(f"Loaded {len(dataset)} episodes")
    if dataset:
        ep = dataset[0]
        print(f"  Episode 0: {ep['meta']['task']}")
        print(f"    actions:  {ep['actions'].shape}")
        if "states" in ep:
            print(f"    states:   {ep['states'].shape}")
        print(f"    rewards:  {ep['rewards'].shape}")
        for key in ep:
            if key.startswith("image_"):
                print(f"    {key}: {ep[key].shape}")
"#;
    fs::write(Path::new(output_dir).join("load_kinodb.py"), loader_script)?;

    println!();
    println!("Done! Exported to {}/", output_dir);
    println!();
    println!("Load in Python:");
    println!("  cd {}", output_dir);
    println!("  python -c \"from load_kinodb import load_dataset; d = load_dataset('.'); print(f'{{len(d)}} episodes')\"");

    Ok(())
}

/// Export just metadata as JSON (no binary data). Quick inspection tool.
fn run_json(kdb_path: &str, output_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    let reader = KdbReader::open(kdb_path)?;

    fs::create_dir_all(output_dir)?;

    let mut episodes_json = Vec::new();
    for i in 0..reader.num_episodes() {
        let meta = reader.read_meta(i)?;
        episodes_json.push(format!(
            concat!(
                "  {{\n",
                "    \"id\": {},\n",
                "    \"embodiment\": \"{}\",\n",
                "    \"task\": \"{}\",\n",
                "    \"num_frames\": {},\n",
                "    \"action_dim\": {},\n",
                "    \"fps\": {},\n",
                "    \"success\": {}\n",
                "  }}",
            ),
            meta.id.0,
            meta.embodiment,
            meta.language_instruction,
            meta.num_frames,
            meta.action_dim,
            meta.fps,
            match meta.success {
                Some(true) => "true",
                Some(false) => "false",
                None => "null",
            },
        ));
    }

    let output = format!(
        "{{\n  \"num_episodes\": {},\n  \"episodes\": [\n{}\n  ]\n}}\n",
        reader.num_episodes(),
        episodes_json.join(",\n"),
    );

    let out_path = Path::new(output_dir).join("episodes.json");
    fs::write(&out_path, output)?;
    println!("Wrote {}", out_path.display());

    Ok(())
}

/// Make a string safe for use as a filename.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
