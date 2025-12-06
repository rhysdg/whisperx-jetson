#!/bin/bash
# Build script adapted from TensorRT-LLM's Whisper example README
# This targets the small.en checkpoint and automates download + build steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
mkdir -p "${ASSETS_DIR}"

MODEL_NAME="${MODEL_NAME:-small.en}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-float16}"
WEIGHT_ONLY_PRECISION="${WEIGHT_ONLY_PRECISION:-int8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_BEAM_WIDTH="${MAX_BEAM_WIDTH:-4}"
CHECKPOINT_DIR="whisper_${MODEL_NAME}_weights_${WEIGHT_ONLY_PRECISION}"
OUTPUT_DIR="whisper_${MODEL_NAME}_${WEIGHT_ONLY_PRECISION}"

# Helper to download only once
fetch_asset() {
    local url="$1"
    local dest="${ASSETS_DIR}/$(basename "$url")"
    if [ ! -f "$dest" ]; then
        echo "Downloading $(basename "$url")..."
        wget --directory-prefix="${ASSETS_DIR}" "$url"
    else
        echo "Skipping download of $(basename "$url"); already present."
    fi
}

fetch_asset https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
fetch_asset https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken
fetch_asset https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client/main/datasets/mini_en/wav/1221-135766-0002.wav
fetch_asset https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt

echo "Converting ${MODEL_NAME} checkpoint into TensorRT-LLM checkpoints..."
python3 convert_checkpoint.py \
    --model_name "${MODEL_NAME}" \
    --use_weight_only \
    --weight_only_precision "${WEIGHT_ONLY_PRECISION}" \
    --output_dir "${CHECKPOINT_DIR}"

# Build encoder
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/encoder" \
    --output_dir "${OUTPUT_DIR}/encoder" \
    --paged_kv_cache disable \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --gemm_plugin disable \
    --bert_attention_plugin "${INFERENCE_PRECISION}" \
    --remove_input_padding disable \
    --max_input_len 3000

# Build decoder
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/decoder" \
    --output_dir "${OUTPUT_DIR}/decoder" \
    --paged_kv_cache disable \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_beam_width "${MAX_BEAM_WIDTH}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_seq_len 114 \
    --max_input_len 14 \
    --max_encoder_input_len 3000 \
    --gemm_plugin "${INFERENCE_PRECISION}" \
    --bert_attention_plugin "${INFERENCE_PRECISION}" \
    --gpt_attention_plugin "${INFERENCE_PRECISION}" \
    --remove_input_padding disable

cat <<EOF

Build complete! Engines saved in ${OUTPUT_DIR}
You can run:
  python3 run.py --engine_dir ${OUTPUT_DIR} --input_file ${ASSETS_DIR}/1221-135766-0002.wav --name small.en
EOF
