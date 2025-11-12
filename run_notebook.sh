#!/usr/bin/env bash
set -euo pipefail

# This script executes a Jupyter notebook non-interactively using papermill.
# It assumes the working directory is the project root (/app in the container).

NOTEBOOK_PATH=${NOTEBOOK_PATH:-loan_meta_optimized.ipynb}
OUTPUT_NOTEBOOK=${OUTPUT_NOTEBOOK:-executed_loan_meta_optimized.ipynb}
PARAMETERS=${PARAMETERS:-{}}
KERNEL_NAME=${KERNEL_NAME:-python3}

# If Data/ and submissions/ exist, list them for visibility
if [ -d "Data" ]; then
  echo "Data directory present:" && ls -lah Data || true
fi
mkdir -p submissions || true

# Install an IPython kernel for papermill if needed (no-op if exists)
python - <<'PY'
import json, sys
try:
    import ipykernel  # noqa: F401
except Exception:
    sys.exit(1)
PY
if [ "$?" -ne 0 ]; then
  pip install ipykernel
fi

# Run the notebook with papermill. If PARAMETERS is JSON, pass it as --parameters_raw.
# Many notebooks don't define tagged parameters; papermill will still execute start->end.

echo "Executing notebook: ${NOTEBOOK_PATH} -> ${OUTPUT_NOTEBOOK}"

papermill "${NOTEBOOK_PATH}" "${OUTPUT_NOTEBOOK}" \
  --kernel "${KERNEL_NAME}" \
  --parameters_raw "${PARAMETERS}" \
  --cwd "/app"

# If a submission was created, list it
if [ -d "submissions" ]; then
  echo "\nSubmissions generated:" && ls -lah submissions || true
fi

# Keep container ephemeral; print summary path
echo "\nNotebook execution complete: ${OUTPUT_NOTEBOOK}"