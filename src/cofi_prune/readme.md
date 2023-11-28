accelerate launch whisper_cloud_hi_cofi.py --max_train_samples 100 --max_test_samples 10 --prepruning_finetune_steps 0 --target_sparsity 0.95 

# --learning_rate 1e-7  -train_batch_size 2 --reg_learning_rate 0.01 --mixed_precision fp16