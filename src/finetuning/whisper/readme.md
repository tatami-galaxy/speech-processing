# Z8

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-tiny --data_dir /home/ujan/Downloads/aishell1 --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-tiny --data_dir /home/ujan/Datasets --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

# change path_remap, batch_size, epochs, csv names between dgx and z8



# 86

rm -r /media/ujan/MHST/dataset

export HF_DATASETS_CACHE="/media/ujan/MHST/dataset"

CUDA_VISIBLE_DEVICES=0,1,2,3 python asr_seq2seq_no_cache.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-medium  --data_dir /home/ujan/Datasets --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end
# --max_train_samples 1000 --max_eval_samples 100 --max_test_samples 100 

CUDA_VISIBLE_DEVICES=1,2,3 python asr_seq2seq_no_cache.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-medium  --data_dir /media/ujan/MHST/far_field --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

# 76
export HF_DATASETS_CACHE="/media/ujan/52850d32-e9c9-47e3-8370-07c2aad12eb1/datasets"



# eval

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-300m --data_dir /home/ujan/Downloads/aishell1 --do_eval ---predict_with_generate --fp16

