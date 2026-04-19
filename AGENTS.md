# AGENTS.md

Guidance for agentic coding tools working in this repository.

## Project Shape

kinodb is a robot trajectory database. The root Rust workspace contains the core storage engine, ingestion layer, CLI, and gRPC server. The Python bindings and documentation site live beside the workspace and are built separately.

```text
crates/
  kinodb-core/    .kdb reader, writer, file layout, KQL, mixtures
  kinodb-ingest/  HDF5, LeRobot, and RLDS importers
  kinodb-cli/     kino command-line interface
  kinodb-serve/   gRPC serving layer
  kinodb-py/      Python bindings; excluded from the root Cargo workspace
kinodb-docs/      Astro Starlight documentation site
assets/           README and project assets
notes/            project notes
```

## Working Rules

- Keep edits scoped to the requested behavior and the affected crate or docs page.
- Do not modify generated build output such as `target/`, `kinodb-docs/dist/`, or compiled Python extension artifacts.
- Preserve the binary names: the CLI binary is `kino`, and the server binary is `kino-serve`.
- The root workspace excludes `crates/kinodb-py`; build Python bindings from that crate with maturin when needed.
- Prefer existing crate boundaries over adding cross-crate dependencies.
- For docs, use the existing Astro Starlight structure under `kinodb-docs/src/content/docs/` and update `kinodb-docs/astro.config.mjs` only when navigation changes.

## Rust Commands

Run these from the repository root unless noted.

```bash
cargo fmt --all
cargo build --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

Useful focused commands:

```bash
cargo test -p kinodb-core
cargo test -p kinodb-ingest
cargo run -p kinodb-cli --bin kino -- --help
cargo run -p kinodb-serve --bin kino-serve -- --help
```

The CI environment installs CMake, HDF5, and `protoc`. Local builds that touch ingestion or serving may need equivalent system packages available.

## Python Bindings

The Python package is separate from the root workspace.

```bash
cd crates/kinodb-py
maturin develop --release
```

The Python source package lives in `crates/kinodb-py/python/kinodb/`. Avoid committing regenerated shared-library artifacts unless the change explicitly requires it.

## Documentation

The docs app is in `kinodb-docs/`.

```bash
cd kinodb-docs
npm install
npm run check
npm run build
npm run dev
```

Docs pages use Markdown or MDX frontmatter with `title` and `description`. Benchmark pages currently include small HTML snippets with project-specific classes from `kinodb-docs/src/styles/custom.css`; keep those class names consistent when extending benchmark visuals.

## File Format And Data Safety

- Treat `.kdb` compatibility carefully. Changes to headers, indexes, episode layout, or reader/writer invariants should include focused tests and docs updates.
- KQL behavior is user-facing. Parser or evaluator changes should include examples or tests around boolean logic, comparisons, and string matching.
- Ingesters should handle partial or malformed source datasets with clear errors rather than panics where practical.
- Avoid checking large generated datasets into the repo. Use synthetic data through the CLI or tests.

## Before Handing Off

For Rust changes, run `cargo fmt --all` and the narrowest relevant `cargo test` command. For workspace-wide or shared behavior, run `cargo test --workspace`; run clippy when changing public APIs, parsing, ingestion, or server code.

For docs-only changes, run `npm run check` and `npm run build` from `kinodb-docs/`.
