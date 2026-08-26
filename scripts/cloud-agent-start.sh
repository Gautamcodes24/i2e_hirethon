#!/usr/bin/env bash
# Sync runtime config on every agent boot (after git checkout may reset .env / LFS files).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Ensuring Git LFS artifacts are materialized"
git lfs pull

GROQ_KEY="${GROQ_API_KEY:-}"
OPENAI_KEY="${OPENAI_API_KEY:-}"

if [[ -f .env ]]; then
  [[ -z "$GROQ_KEY" ]] && GROQ_KEY="$(grep -m1 '^GROQ_API_KEY=' .env | cut -d= -f2- || true)"
  [[ -z "$OPENAI_KEY" ]] && OPENAI_KEY="$(grep -m1 '^OPENAI_API_KEY=' .env | cut -d= -f2- || true)"
fi

if [[ -n "$GROQ_KEY" || -n "$OPENAI_KEY" ]]; then
  cat > .env <<EOF
GROQ_API_KEY=${GROQ_KEY}
OPENAI_API_KEY=${OPENAI_KEY}
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=groq
VISION_PROVIDER=groq
EOF
  echo "==> Wrote .env with provider settings"
fi
