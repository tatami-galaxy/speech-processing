accelerate launch whisper_distil.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-small --teacher_name_or_path /home/ujan/speech-processing/models/whisper/whisper-medium_Datasets/checkpoint-27000 --data_dir /home/ujan/Datasets --max_train_samples 1000 --max_test_samples 100 --max_eval_samples 100
# --gradient_accumulation_steps 4 --resume_from_checkpoint "checkpoint" --skip_steps --max_train_samples 1000 --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --preprocessing_num_workers 1

Version 2(without aishell4 hkust):
final_train_v2a_wo_aishell4_hkust.csv
final_dev_v2a_wo_aishell4_hkust.csv
final_test_v2a_wo_aishell4_hkust.csv
