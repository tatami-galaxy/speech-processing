accelerate launch whisper_cofi_hi.py --prepruning_finetune_steps 0 --lagrangian_warmup_epochs 0 --target_sparsity 0.95 

# --data_dir mozilla-foundation/common_voice_11_0
# --max_train_samples 100 --max_test_samples 10 --eval_steps 5 
# --learning_rate 1e-7  --train_batch_size 2 --reg_learning_rate 0.01 --mixed_precision fp16
