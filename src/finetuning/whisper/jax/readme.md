python whisper_flax_torch_loader.py --dtype float32 --overwrite_output_dir
python whisper_flax.py --dtype float32 --overwrite_output_dir
# --dtype float32 for cpu
# --dtype float16 --max_train_samples 100 --max_test_samples 20

# for linking error
export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 

