#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/generated"
TMP_DIR="${TMPDIR:-/tmp}/speechkit-cloudapi"

rm -rf "$TMP_DIR"
git clone --depth=1 https://github.com/yandex-cloud/cloudapi "$TMP_DIR"
mkdir -p "$OUT_DIR"
cd "$TMP_DIR"
python -m grpc_tools.protoc -I . -I third_party/googleapis \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  google/api/http.proto \
  google/api/annotations.proto \
  yandex/cloud/api/operation.proto \
  google/rpc/status.proto \
  yandex/cloud/operation/operation.proto \
  yandex/cloud/validation.proto \
  yandex/cloud/ai/stt/v3/stt_service.proto \
  yandex/cloud/ai/stt/v3/stt.proto \
  yandex/cloud/ai/tts/v3/tts_service.proto \
  yandex/cloud/ai/tts/v3/tts.proto

echo "Generated SpeechKit gRPC stubs into $OUT_DIR"
