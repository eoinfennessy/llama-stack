#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Generate Python gRPC stubs from proto definitions.
# Requires: pip install grpcio-tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PROTOS_DIR="$REPO_ROOT/src/llama_stack_api/inference/grpc/protos"
OUT_DIR="$REPO_ROOT/src/llama_stack_api/inference/grpc/generated"

mkdir -p "$OUT_DIR"

uv run python -m grpc_tools.protoc \
  --proto_path="$PROTOS_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  --pyi_out="$OUT_DIR" \
  "$PROTOS_DIR/inference.proto"

# Fix imports in generated gRPC file to use relative imports.
# Using a temp file avoids macOS/Linux sed portability issues.
tmp="$OUT_DIR/inference_pb2_grpc.py.tmp"
sed 's/^import inference_pb2/from . import inference_pb2/' "$OUT_DIR/inference_pb2_grpc.py" > "$tmp"
mv "$tmp" "$OUT_DIR/inference_pb2_grpc.py"

echo "Generated gRPC stubs in $OUT_DIR"
