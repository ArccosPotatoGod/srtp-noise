#!/usr/bin/env bash
# Wrapper script to download/prepare audio data
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$PROJECT_ROOT/venv/bin/activate" 2>/dev/null || true
python "$PROJECT_ROOT/scripts/download_data.py" "$@"
