Cloud

# accelerate launch whisper_cofi.py --model_name_or_path openai/whisper-small --prepruning_finetune_steps 2000 --train_steps 7000 --eval_steps 1000 --lagrangian_warmup_steps 0 --target_sparsity 0.95

# lagrangian_warmup_steps 200 --teacher_name_or_path `teacher path` --distil_type logit --max_test_samples 300

# --learning_rate 1e-7 --reg_learning_rate 0.01 --mixed_precision fp16

Local

# export HF_DATASETS_CACHE="/media/ujan/MHST/RA/datasets"

# accelerate launch whisper_cofi.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-small  --prepruning_finetune_steps 2000 --train_steps 10000 --eval_steps 1000 --lagrangian_warmup_steps 200 --target_sparsity 0.95 --data_dir /media/ujan/MHST/RA/datasets/common_voice_13_hi --teacher_name_or_path /home/ujan/speech-processing/models/whisper/whisper-medium-cv-13-hi-6000 --distil_type logit --max_test_samples 300 --num_workers 64 --local