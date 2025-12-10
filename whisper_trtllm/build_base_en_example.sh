#!/bin/bash
# Build script for base.en model optimized for CLI conversation
# Low memory footprint: batch=1, beam=1, paged KV cache
# MAX_INPUT_LEN=3000 is REQUIRED by Whisper architecture (30 seconds @ 100Hz mel spectrogram)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
mkdir -p "${ASSETS_DIR}"

MODEL_NAME="${MODEL_NAME:-base.en}"
INFERENCE_PRECISION="${INFERENCE_PRECISION:-float16}"
WEIGHT_ONLY_PRECISION="${WEIGHT_ONLY_PRECISION:-int8}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"  # CLI: process one utterance at a time
MAX_BEAM_WIDTH="${MAX_BEAM_WIDTH:-1}"  # CLI: greedy decoding saves memory
MAX_INPUT_LEN="${MAX_INPUT_LEN:-3000}"  # 30 seconds @ 100Hz (Whisper architecture requirement)
CHECKPOINT_DIR="whisper_${MODEL_NAME}_weights_${WEIGHT_ONLY_PRECISION}"
OUTPUT_DIR="whisper_${MODEL_NAME}"

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
fetch_asset https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt

echo "Converting ${MODEL_NAME} checkpoint into TensorRT-LLM checkpoints..."
python3 convert_checkpoint.py \
    --model_name "${MODEL_NAME}" \
    --use_weight_only \
    --weight_only_precision "${WEIGHT_ONLY_PRECISION}" \
    --output_dir "${CHECKPOINT_DIR}"

# Build encoder
echo "Building encoder with max_input_len=${MAX_INPUT_LEN} (~$((MAX_INPUT_LEN / 100)) seconds)..."
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/encoder" \
    --output_dir "${OUTPUT_DIR}/encoder" \
    --paged_kv_cache enable \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --gemm_plugin disable \
    --bert_attention_plugin "${INFERENCE_PRECISION}" \
    --remove_input_padding disable \
    --max_input_len "${MAX_INPUT_LEN}"

# Build decoder
echo "Building decoder with batch=${MAX_BATCH_SIZE}, beam=${MAX_BEAM_WIDTH}..."
trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}/decoder" \
    --output_dir "${OUTPUT_DIR}/decoder" \
    --paged_kv_cache enable \
    --moe_plugin disable \
    --enable_xqa disable \
    --max_beam_width "${MAX_BEAM_WIDTH}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_seq_len 114 \
    --max_input_len 14 \
    --max_encoder_input_len "${MAX_INPUT_LEN}" \
    --gemm_plugin "${INFERENCE_PRECISION}" \
    --bert_attention_plugin "${INFERENCE_PRECISION}" \
    --gpt_attention_plugin "${INFERENCE_PRECISION}" \
    --remove_input_padding disable

cat <<EOF

✅ Build complete! Engines saved in ${OUTPUT_DIR}

Configuration:
  Model: ${MODEL_NAME}
  Precision: ${WEIGHT_ONLY_PRECISION}
  Max batch size: ${MAX_BATCH_SIZE}
  Max beam width: ${MAX_BEAM_WIDTH}
  Max input length: ${MAX_INPUT_LEN} frames (~$((MAX_INPUT_LEN / 100)) seconds)
  Paged KV cache: enabled

Test with:
  python3 run.py --engine_dir ${OUTPUT_DIR} --input_file ${ASSETS_DIR}/1221-135766-0002.wav --name base.en --max_input_len ${MAX_INPUT_LEN} --padding_strategy max

Use in CLI agent:
  python3 cli_agent.py --speech --asr-backend tensorrt-llm --asr-engine-dir ${OUTPUT_DIR}
EOF
