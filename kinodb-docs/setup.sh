#!/bin/bash
# Setup script: merges markdown docs into Starlight structure
# Run this after extracting both kinodb_docs.tar.gz and kinodb-starlight-docs.tar.gz

# If kinodb_docs/ exists alongside, copy content into Starlight
DOCS_SRC="${1:-../kinodb_docs/docs}"

if [ -d "$DOCS_SRC" ]; then
    echo "Copying docs from $DOCS_SRC into Starlight..."
    
    # Map old filenames to new locations
    cp_with_frontmatter() {
        local src="$1" dst="$2" title="$3"
        if [ ! -f "$src" ]; then echo "  SKIP: $src not found"; return; fi
        echo "---" > "$dst"
        echo "title: \"$title\"" >> "$dst"
        echo "---" >> "$dst"
        echo "" >> "$dst"
        # Skip the first line (# Title) from the source
        tail -n +2 "$src" >> "$dst"
        echo "  OK: $dst"
    }
    
    cp_with_frontmatter "$DOCS_SRC/getting-started.md" "src/content/docs/guides/quickstart.md" "Quick Start"
    cp_with_frontmatter "$DOCS_SRC/cli-reference.md" "src/content/docs/reference/cli.md" "CLI Reference"
    cp_with_frontmatter "$DOCS_SRC/python-api.md" "src/content/docs/reference/python-api.md" "Python API"
    cp_with_frontmatter "$DOCS_SRC/file-format.md" "src/content/docs/reference/file-format.md" "File Format Specification"
    cp_with_frontmatter "$DOCS_SRC/architecture.md" "src/content/docs/reference/architecture.md" "Architecture"
    cp_with_frontmatter "$DOCS_SRC/ingestion-guide.md" "src/content/docs/guides/ingestion.md" "Ingesting Data"
    cp_with_frontmatter "$DOCS_SRC/pytorch-integration.md" "src/content/docs/guides/pytorch.md" "PyTorch Integration"
    cp_with_frontmatter "$DOCS_SRC/kql.md" "src/content/docs/guides/kql.md" "KQL Query Language"
    cp_with_frontmatter "$DOCS_SRC/grpc-serving.md" "src/content/docs/guides/remote.md" "Remote Serving"
    cp_with_frontmatter "$DOCS_SRC/benchmarks.md" "src/content/docs/benchmarks/io.md" "IO Performance"
    
    echo "Done!"
else
    echo "Docs source not found at $DOCS_SRC"
    echo "Usage: bash setup.sh /path/to/kinodb_docs/docs"
fi
