# choose the engine you build [./whisper_large_v3, ./whisper_large_v3_int8]
output_dir=./whisper_large_v3_int8
# decode a single audio file
# If the input file does not have a .wav extension, ffmpeg needs to be installed with the following command:
# apt-get update && apt-get install -y ffmpeg
# Inferencing via python binding of C++ runtime with inflight batching (IFB)
python3 run.py --name single_wav_test --engine_dir $output_dir --input_file test.wav --enable_warmup


#decode a whole dataset
#python3 run.py --engine_dir $output_dir --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup --name librispeech_dummy_large_v3
