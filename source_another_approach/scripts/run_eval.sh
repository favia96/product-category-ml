#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m shop_ml.eval --data_dir ./dataset --ckpt ./artifacts/best_model.pt --out_dir ./artifacts "$@"
