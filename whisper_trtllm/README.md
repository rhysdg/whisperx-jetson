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

After the script completes, run inference using `run.py`:

```bash
python3 run.py --engine_dir ./whisper_small.en_int8 --input_file assets/1221-135766-0002.wav --name small.en
```

This example copies the official TensorRT-LLM 0.12 Whisper entry point, which expects 3000-frame mel inputs. If you run into the `Static dimension mismatch` error, add `--max_input_len 3000 --padding_strategy max` (the default) so shorter WAV files are padded or longer ones trimmed before decoding. You can choose `--padding_strategy trim` to only trim longer inputs without padding.

If you hit CUDA pinned-memory limits on Jetson-class hardware, rerun with a smaller generation budget, for example `--max_new_tokens 64`. That trims the decoder’s attention window and keeps `DynamicDecodeOp` from allocating extra buffers.
