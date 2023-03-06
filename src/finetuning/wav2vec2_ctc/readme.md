python asr_ctc.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-300m --data_dir /home/ujan/Downloads/aishell1 --freeze_feature_encoder --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --group_by_length --fp16 --load_best_model_at_end

change path_remap, batch_size, epochs, csv names between dgx and z8


# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python asr_ctc.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-1b --data_dir /home/ujan/Datasets --preprocessing_num_workers 128 --freeze_feature_encoder --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --group_by_length --load_best_model_at_end --fp16

# eval
CUDA_VISIBLE_DEVICES=0,1,2,3 python asr_ctc.py --model_name_or_path /home/ujan/speech-processing/models/wav2vec2/wav2vec2-xls-r-1b_Datasets/checkpoint-33000 --output_dir /home/ujan/speech-processing/models/wav2vec2/wav2vec2-xls-r-1b_Datasets/ --data_dir /home/ujan/Datasets --preprocessing_num_workers 128 --max_train_samples 100 --max_eval_samples 100 --freeze_feature_encoder --do_eval --gradient_checkpointing --group_by_length --load_best_model_at_end --fp16