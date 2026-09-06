#!/usr/bin/env bash
set -e

echo "=== Building Enterprise CUDA Image Pipeline ==="
make clean
make all

echo "=== Executing Batch Image Pipeline (120 Images) ==="
./bin/batch_image_pipeline --input-dir data/input --output-dir data/output --streams 4 --threshold 75 | tee logs/execution_log.txt

echo "=== Pipeline Completed Successfully ==="
