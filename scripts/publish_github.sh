#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-enterprise-memory-mlx}"
VISIBILITY="${2:-private}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI first: brew install gh" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "Authenticate GitHub CLI first: gh auth login" >&2
  exit 1
fi
if [[ "$VISIBILITY" != "private" && "$VISIBILITY" != "public" ]]; then
  echo "Visibility must be private or public." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init -b main
  git add .
  git commit -m "Initial MLX enterprise memory experiment"
fi

gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
