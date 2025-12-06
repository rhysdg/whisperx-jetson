# Patch Notes

- Ensure generated encoder/decoder configs now expose the `n_audio_*` and `n_text_*` metadata so TensorRT-LLM's `PretrainedConfig` doesn't need to guess those attributes and can build without `AttributeError`.
- Save the `positional_embedding` tensor in addition to `position_embedding.weight` so TensorRT-LLM sees the tensor it expects when loading checkpoints.
- Documented the small.en build workflow in `README.md` and added `build_small_en_example.sh` so you can reproduce the example steps with a dedicated script.
- Replaced `run.py` with the current official TensorRT-LLM 0.12 Whisper example so the runtime imports and session logic match the packaged `tensorrt_llm` wheel.
- Added CLI flags for `--max_input_len`/`--padding_strategy` so mel lengths can be padded/truncated to match the 0.12 encoder profile and avoid the static dimension mismatch error.
- Added a `--max_new_tokens` CLI flag so Jetson users can dial down the decoder budget when `DynamicDecodeOp` hits pinned-host memory limits.
