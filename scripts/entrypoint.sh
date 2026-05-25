#!/bin/bash
set -e

if [ -n "$GCS_INDEX_URI" ]; then
  echo "Downloading index from $GCS_INDEX_URI..."
  mkdir -p /tmp/index
  python3 scripts/gcs_download.py
  export INDEX_PATH=/tmp/index
fi

exec uvicorn src.service.main:app --host 0.0.0.0 --port ${PORT:-8000}
