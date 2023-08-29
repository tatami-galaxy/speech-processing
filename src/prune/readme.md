accelerate launch whisper_cv_prune.py 
# --pruning_method sigmoied_threshold --regularization l1 --final_lambda 400 --initial_threshold 0 --final_threshold 0.1 --max_test_samples 50 --warmup_steps 1000 --train_steps 10000 
 
python paste_mask.py --pruning_method sigmoied_threshold --threshold 0.0783783927992 --sparsity_threshold 0.7 --block_size 32 --model_name_or_path /home/ujan/speech-processing/models/whisper/whisper-small-common_voice_13_0-sigmoied_threshold-pruned/checkpoint-2000