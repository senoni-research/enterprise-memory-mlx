#!/usr/bin/env bash
set -euo pipefail

echo "Disabled: this script invokes the scientifically invalid legacy pipeline." >&2
echo "Use 'emmlx benchmark --dry-run' or 'emmlx review' instead." >&2
exit 2
