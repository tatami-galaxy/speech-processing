accelerate launch whisper_distil.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-small --teacher_name_or_path "../teacher" --data_dir /home/ujan/Datasets
# --gradient_accumulation_steps 4 --resume_from_checkpoint "checkpoint" --skip_steps --max_train_samples 1000 --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --preprocessing_num_workers 1

