accelerate launch whisper_cofi_hi.py --prepruning_finetune_steps 0 --target_sparsity 0.95 

# --max_train_samples 100 --max_test_samples 10 --lagrangian_warmup_epochs 0 --eval_steps 5 
# --learning_rate 1e-7  --train_batch_size 2 --reg_learning_rate 0.01 --mixed_precision fp16