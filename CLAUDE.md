# CLAUDE.md

Claude Code and other Claude-based tools should use `AGENTS.md` as the canonical project guide. Read that file before making edits.

Quick reference:

- Rust workspace crates live under `crates/`, except `crates/kinodb-py`, which is built separately with maturin.
- The CLI binary is `kino`; the gRPC server binary is `kino-serve`.
- Documentation lives in `kinodb-docs/` and is an Astro Starlight site.
- Avoid generated output: `target/`, `kinodb-docs/dist/`, and compiled Python extension artifacts.

Common checks:

```bash
cargo fmt --all
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

Docs checks:

```bash
cd kinodb-docs
npm run check
npm run build
```

Python binding build:

```bash
cd crates/kinodb-py
maturin develop --release
```
