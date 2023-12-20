accelerate launch whisper_cofi.py --model_name_or_path openai/whisper-small --prepruning_finetune_steps 2000 --train_steps 7000 --eval_steps 1000 --lagrangian_warmup_steps 0 --target_sparsity 0.95

# lagrangian_warmup_steps 200 --teacher_name_or_path `teacher path` --distil_type logit --local

# --learning_rate 1e-7 --reg_learning_rate 0.01 --mixed_precision fp16
