#!/usr/bin/env bash
set -euo pipefail

BATCH_ID="${1:-tau3voice_gptrt2_stagegate_$(date +%Y_%m_%d)}"
REPO_REF="${2:-$(git rev-parse HEAD)}"

printf 'Launching Modal StageGate matrix\n'
printf 'Batch ID: %s\n' "$BATCH_ID"
printf 'Repo ref: %s\n' "$REPO_REF"

modal run modal_tau3_voice_stagegate.py \
  --batch-id "$BATCH_ID" \
  --repo-ref "$REPO_REF"
