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
  echo "Wrote .env with provider settings"
elif [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Copied .env.example to .env (add API keys in Environment settings)"
fi

echo "==> Install complete"
