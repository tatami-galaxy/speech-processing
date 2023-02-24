python asr_ctc.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-300m --data_dir /home/ujan/Downloads/aishell1 --learning_rate 3e-4 --lr_scheduler_type constant --max_train_samples 1000 --max_eval_samples 100 --max_test_samples 100 --eval_steps 100 --save_steps 100 --logging_steps 100 --warmup_steps 0 --freeze_feature_encoder --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --group_by_length --load_best_model_at_end

change path_remap, batch_size, epochs, csv names between dgx and z8


## run this ##
## from /home/ujan/speech-processing/src/finetuning/wav2vec2_ctc in DGX ##
## use tmux ##

CUDA_VISIBLE_DEVICES=2,3 python asr_ctc.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-1b --data_dir /home/ujan/Datasets --preprocessing_num_workers 128 --freeze_feature_encoder --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --group_by_length --load_best_model_at_end

CUDA_VISIBLE_DEVICES=0,1,2,3 python asr_ctc.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-1b --data_dir /home/ujan/Datasets --max_train_samples 10000 --max_eval_samples 1000 --max_test_samples 1000 --eval_steps 100 --save_steps 100 --logging_steps 100 --warmup_steps 0 --preprocessing_num_workers 128 --freeze_feature_encoder --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --group_by_length --load_best_model_at_end --fp16