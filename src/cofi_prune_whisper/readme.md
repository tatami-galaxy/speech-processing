export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

### Cloud

accelerate launch whisper_cofi.py --model_name_or_path openai/whisper-small --prepruning_finetune_steps 2000 --train_steps 8000 --eval_steps 1000 --target_sparsity 0.96 --max_test_samples 300 --minimize_mask_entropy --lagrangian_warmup_steps 200 --teacher_name_or_path `teacher path` --distil_type rail --rail_steps 2000 --max_test_samples 300 --reg_learning_rate 0.02 --l0_temperature 0.03

accelerate launch whisper_cofi.py --model_name_or_path openai/whisper-small --prepruning_finetune_steps 4000 --train_steps 10000 --eval_steps 1000 --lagrangian_warmup_steps 0 --target_sparsity 0.96 --max_test_samples 300 --teacher_name_or_path Ujan/whisper-medium-cv-13-hi-6000 --distil_type rail --rail_steps 2000 --reg_learning_rate 0.02 --l0_temperature 0.03 --train_batch_size 2 --eval_batch_size 2 --model_lang bengali --data_lang bn

### Local

export HF_DATASETS_CACHE="/media/ujan/asr/datasets"

accelerate launch whisper_cofi.py --model_name_or_path /media/ujan/asr/speech-processing/models/pretrained_models/whisper-small --prepruning_finetune_steps 2000 --train_steps 10000 --eval_steps 2000 --lagrangian_warmup_steps 200 --target_sparsity 0.96 --data_dir /media/ujan/MHST/RA/datasets/common_voice_13_hi --teacher_name_or_path /media/ujan/asr/speech-processing/models/whisper/whisper-medium-cv-13-hi-6000 --distil_type rail --max_test_samples 300 --num_workers 64 --rail_steps 2000 --local

--ldc

accelerate launch whisper_cofi.py --model_name_or_path /media/ujan/asr/speech-processing/models/pretrained_models/whisper-small --prepruning_finetune_steps 6000 --train_steps 30000 --eval_steps 2000 --lagrangian_warmup_steps 2000 --target_sparsity 0.96 --l0_temperature 0.02 --reg_learning_rate 0.01 --data_dir /home/ujan/Datasets --output_dir /media/ujan/asr/whisper/whisper-small-cofi --teacher_name_or_path /media/ujan/asr/speech-processing/models/whisper/whisper-medium_Datasets/checkpoint-46000 --distil_type rail --max_test_samples 1000 --num_workers 64 --rail_steps 2000 --ent_const 5 --ldc
