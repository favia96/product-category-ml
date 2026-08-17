#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python -m shop_ml.train --data_dir ./dataset --out_dir ./artifacts "$@"
