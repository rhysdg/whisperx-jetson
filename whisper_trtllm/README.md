# whisper_trtllm

## Dependencies

Install the Python packages needed to run the included scripts and engines. Install them with the `requirements.txt` from this directory.

```bash
cd ~/mr-b/src/whisper_trtllm
python3 -m pip install -r requirements.txt
```

This directory now ships the official TensorRT-LLM 0.12 Whisper example `run.py` so you can drop straight into the runtime that ships with the upstream wheels. The helper utilities (`whisper_utils.py`, `tokenizer.py`, etc.) remain the same, but the inference flow now mirrors the current sample from `TensorRT-LLM/examples/whisper`.

## Example build script for `small.en`

`build_small_en_example.sh` mirrors the TensorRT-LLM Whisper example README but targets the compact `small.en` checkpoint and automates downloading the necessary assets. It:

1. Downloads `mel_filters.npz`, `gpt2.tiktoken`, a sample wav file, and the `small.en.pt` model.
2. Converts the Whisper checkpoint into TensorRT-LLM weight-only checkpoints.
3. Builds encoder and decoder engines with reasonable defaults for the tiny model.

### Usage

```bash
cd ~/mr-b/src/whisper_trtllm
./build_small_en_example.sh
```

You can override the defaults with environment variables, for example:

```bash
MODEL_NAME=small.en \ 
WEIGHT_ONLY_PRECISION=int8 \ 
MAX_BATCH_SIZE=4 \ 
./build_small_en_example.sh
```

## Ultra-Low Memory Build: `base.en` for Duplex Conversations

`build_base_en_example.sh` provides an **ultra-low memory** configuration optimized for running ASR + TTS simultaneously on memory-constrained devices (e.g., Jetson Orin Nano 8GB).

**Key differences from `small.en`**:
- Uses smaller `base.en` model (~74M params vs ~244M params)
- Defaults to `MAX_INPUT_LEN=750` (~7.5 seconds per turn)
- `MAX_BATCH_SIZE=1` and `MAX_BEAM_WIDTH=1` (greedy decoding)
- Paged KV cache enabled by default

**When to use**:
- Running ASR alongside TensorRT TTS models (e.g., KokoroTRT)
- Memory-constrained devices (< 8GB)
- Fast duplex conversations where both models need to stay loaded

### Usage

```bash
cd ~/mr-b/src/whisper_trtllm
./build_base_en_example.sh
```

Override for longer utterances:

```bash
MAX_INPUT_LEN=1500 ./build_base_en_example.sh  # ~15 seconds per turn
```

### Running the Built Engines

After the script completes, run inference using `run.py`:

```bash
python3 run.py --engine_dir ./whisper_small.en_int8 --input_file assets/1221-135766-0002.wav --name small.en
```

This example copies the official TensorRT-LLM 0.12 Whisper entry point, which expects 3000-frame mel inputs. If you run into the `Static dimension mismatch` error, add `--max_input_len 3000 --padding_strategy max` (the default) so shorter WAV files are padded or longer ones trimmed before decoding. You can choose `--padding_strategy trim` to only trim longer inputs without padding.

If you hit CUDA pinned-memory limits on Jetson-class hardware, rerun with a smaller generation budget, for example `--max_new_tokens 64`. That trims the decoder's attention window and keeps `DynamicDecodeOp` from allocating extra buffers.

## Build Configuration for Different Use Cases

The default build parameters in `build_small_en_example.sh` can be tuned based on your deployment scenario:

### Interactive CLI Conversations (Recommended for Low Memory)

For real-time voice conversations where you process one utterance at a time:

```bash
MAX_BATCH_SIZE=1 \
MAX_BEAM_WIDTH=1 \
MAX_INPUT_LEN=1500 \
./build_small_en_example.sh
```

**Why these settings?**
- **`MAX_BATCH_SIZE=1`**: CLI conversations are turn-based (you speak → pause → wait for response). Never need to process multiple speakers simultaneously.
- **`MAX_BEAM_WIDTH=1`**: Greedy decoding is nearly as accurate as beam search for Whisper (< 1% WER difference) and significantly faster with less memory usage.
- **`MAX_INPUT_LEN=1500`**: ~15 seconds of audio per turn. Natural conversational utterances are typically 3-10 seconds, giving comfortable buffer.

**Memory Impact**: Reducing from default `batch=8, beam=4` to `batch=1, beam=1` cuts KV cache allocation by ~97% (32 → 1 slot), crucial for running alongside other models like TTS on memory-constrained devices.

**Quality**: Beam width of 1 vs 4-5 shows minimal difference (< 0.5-1% WER) in practice. For interactive applications, the latency improvement is more valuable than marginal accuracy gains.

### Batch Transcription (Higher Throughput)

For offline processing of multiple audio files or longer recordings:

```bash
MAX_BATCH_SIZE=8 \
MAX_BEAM_WIDTH=4 \
MAX_INPUT_LEN=3000 \
./build_small_en_example.sh
```

Use higher batch sizes and beam widths when:
- Processing multiple files in parallel
- Accuracy is more critical than latency
- Memory constraints are less severe

### Memory Optimization: Paged KV Cache

**Enable paged KV cache** for better memory management, especially when running multiple models:

```bash
# Edit build_small_en_example.sh and change:
--paged_kv_cache disable  →  --paged_kv_cache enable
```

**Benefits**:
- Allocates memory in smaller pages instead of large contiguous blocks
- Reduces memory fragmentation
- Enables better coexistence with other TensorRT models (e.g., TTS decoders)
- Dynamic allocation based on actual sequence length vs worst-case pre-allocation

**When to use**: Always recommended for Jetson devices with limited memory (< 16GB), especially when running ASR + TTS in duplex mode.
