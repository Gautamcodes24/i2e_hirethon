#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Pulling Git LFS data artifacts"
git lfs pull

echo "==> Creating Python virtual environment"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

echo "==> Installing Python dependencies"
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q

echo "==> Configuring environment variables"
"$(dirname "$0")/cloud-agent-start.sh"

echo "==> Install complete"
