python whisper_flax.py --dtype float32 
python whisper_flax_torch_loader.py --dtype float32 
# --model_name_or_path openai/whisper-tiny
# --overwrite_output_dir
# --dtype float32 for cpu
# --dtype float16
# --max_train_samples 100
# --max_test_samples 20
# --per_device_train_batch_size 8

python whisper_flax.py --dtype float32 --model_name_or_path openai/whisper-tiny --max_train_samples 100 --max_test_samples 20 --overwrite_output_dir

# eval
python whisper_flax_eval.py --dtype float32 --model_name_or_path " " --checkpoint_dir " " --model_lang chinese --data_lang zh-CN

# runpod
python whisper_flax.py --model_name_or_path openai/whisper-small --per_device_train_batch_size 8 --overwrite_output_dir --num_workers 1

# for linking error
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 

