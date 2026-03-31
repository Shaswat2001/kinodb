use kinodb_core::KdbReader;

pub fn run(kdb_path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating: {}", kdb_path);
    println!();

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // ── 1. Open and parse header ────────────────────────────
    let reader = match KdbReader::open(kdb_path) {
        Ok(r) => r,
        Err(e) => {
            println!("FATAL: cannot open file: {}", e);
            println!();
            println!("Result: FAILED (1 fatal error)");
            return Ok(());
        }
    };

    let header = reader.header();
    if verbose {
        println!(
            "  [ok] header: magic=KINO version={}.{}",
            header.version_major, header.version_minor
        );
    }

    // ── 2. Check episode count consistency ──────────────────
    if header.num_episodes as usize != reader.num_episodes() {
        errors.push(format!(
            "header.num_episodes ({}) != index length ({})",
            header.num_episodes,
            reader.num_episodes()
        ));
    } else if verbose {
        println!(
            "  [ok] episode count: {} in header, {} in index",
            header.num_episodes,
            reader.num_episodes()
        );
    }

    // ── 3. Check total frame count ──────────────────────────
    let mut computed_frames: u64 = 0;
    let index = reader.index();
    for i in 0..reader.num_episodes() {
        if let Some(entry) = index.get(i) {
            computed_frames += entry.num_frames as u64;
        }
    }
    if header.num_frames != computed_frames {
        errors.push(format!(
            "header.num_frames ({}) != sum of index entries ({})",
            header.num_frames, computed_frames
        ));
    } else if verbose {
        println!("  [ok] frame count: {} total", header.num_frames);
    }

    // ── 4. Check each episode ───────────────────────────────
    let mut episodes_checked = 0u64;
    let mut frames_checked = 0u64;
    let mut images_checked = 0u64;

    for i in 0..reader.num_episodes() {
        let entry = match index.get(i) {
            Some(e) => e,
            None => {
                errors.push(format!("episode {}: missing index entry", i));
                continue;
            }
        };

        // Check episode_id is sequential
        if entry.episode_id.0 != i as u64 {
            warnings.push(format!(
                "episode {}: id={} (expected {})",
                i, entry.episode_id.0, i
            ));
        }

        // Try reading metadata
        match reader.read_meta(i) {
            Ok(meta) => {
                // Check metadata consistency with index
                if meta.num_frames != entry.num_frames {
                    errors.push(format!(
                        "episode {}: meta.num_frames ({}) != index.num_frames ({})",
                        i, meta.num_frames, entry.num_frames
                    ));
                }
                if meta.action_dim != entry.action_dim {
                    errors.push(format!(
                        "episode {}: meta.action_dim ({}) != index.action_dim ({})",
                        i, meta.action_dim, entry.action_dim
                    ));
                }

                // Check for empty strings
                if meta.embodiment.is_empty() {
                    warnings.push(format!("episode {}: empty embodiment string", i));
                }
                if meta.language_instruction.is_empty() {
                    warnings.push(format!("episode {}: empty task string", i));
                }
            }
            Err(e) => {
                errors.push(format!("episode {}: failed to read metadata: {}", i, e));
                continue;
            }
        }

        // Try reading full episode
        match reader.read_episode(i) {
            Ok(ep) => {
                let n = ep.frames.len();

                // Frame count matches
                if n != entry.num_frames as usize {
                    errors.push(format!(
                        "episode {}: decoded {} frames, index says {}",
                        i, n, entry.num_frames
                    ));
                }

                // Check each frame
                for (t, frame) in ep.frames.iter().enumerate() {
                    // Action dimension
                    if frame.action.len() != entry.action_dim as usize {
                        errors.push(format!(
                            "episode {} frame {}: action.len()={} != action_dim={}",
                            i,
                            t,
                            frame.action.len(),
                            entry.action_dim
                        ));
                    }

                    // State dimension consistency
                    if t > 0 && frame.state.len() != ep.frames[0].state.len() {
                        errors.push(format!(
                            "episode {} frame {}: state.len()={} != frame0.state.len()={}",
                            i,
                            t,
                            frame.state.len(),
                            ep.frames[0].state.len()
                        ));
                    }

                    // NaN/Inf check on actions
                    for (d, &val) in frame.action.iter().enumerate() {
                        if val.is_nan() || val.is_infinite() {
                            warnings.push(format!(
                                "episode {} frame {} action[{}]: {}",
                                i,
                                t,
                                d,
                                if val.is_nan() { "NaN" } else { "Inf" }
                            ));
                        }
                    }

                    // NaN/Inf check on state
                    for (d, &val) in frame.state.iter().enumerate() {
                        if val.is_nan() || val.is_infinite() {
                            warnings.push(format!(
                                "episode {} frame {} state[{}]: {}",
                                i,
                                t,
                                d,
                                if val.is_nan() { "NaN" } else { "Inf" }
                            ));
                        }
                    }

                    // Image validation
                    for img in &frame.images {
                        let expected =
                            (img.width as usize) * (img.height as usize) * (img.channels as usize);
                        if img.data.len() != expected {
                            errors.push(format!(
                                "episode {} frame {} camera '{}': data.len()={} != {}x{}x{}={}",
                                i,
                                t,
                                img.camera,
                                img.data.len(),
                                img.width,
                                img.height,
                                img.channels,
                                expected
                            ));
                        }
                        images_checked += 1;
                    }

                    frames_checked += 1;
                }

                // Last frame should be terminal
                if n > 0 && !ep.frames[n - 1].is_terminal {
                    warnings.push(format!("episode {}: last frame is not terminal", i));
                }
            }
            Err(e) => {
                errors.push(format!("episode {}: failed to read: {}", i, e));
            }
        }

        episodes_checked += 1;

        // Progress for large datasets
        if reader.num_episodes() >= 100 && (i + 1) % 100 == 0 {
            println!("  checked {}/{} episodes...", i + 1, reader.num_episodes());
        }
    }

    // ── Report ──────────────────────────────────────────────
    println!();
    println!("Checked:");
    println!("  {} episodes", episodes_checked);
    println!("  {} frames", frames_checked);
    println!("  {} images", images_checked);
    println!();

    if !warnings.is_empty() {
        println!("Warnings ({}):", warnings.len());
        let show = if verbose {
            warnings.len()
        } else {
            std::cmp::min(10, warnings.len())
        };
        for w in warnings.iter().take(show) {
            println!("  ⚠ {}", w);
        }
        if !verbose && warnings.len() > 10 {
            println!(
                "  ... and {} more (use --verbose to see all)",
                warnings.len() - 10
            );
        }
        println!();
    }

    if !errors.is_empty() {
        println!("Errors ({}):", errors.len());
        let show = if verbose {
            errors.len()
        } else {
            std::cmp::min(10, errors.len())
        };
        for e in errors.iter().take(show) {
            println!("  ✗ {}", e);
        }
        if !verbose && errors.len() > 10 {
            println!(
                "  ... and {} more (use --verbose to see all)",
                errors.len() - 10
            );
        }
        println!();
        println!(
            "Result: FAILED ({} errors, {} warnings)",
            errors.len(),
            warnings.len()
        );
    } else if !warnings.is_empty() {
        println!("Result: PASSED with {} warnings", warnings.len());
    } else {
        println!("Result: PASSED ✓");
    }

    Ok(())
}
