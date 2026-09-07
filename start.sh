#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

exec venv/bin/uvicorn theseus_insight.main:app \
  --env-file .env \
  --host 127.0.0.1 \
  --port 8000
