# Z8

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-tiny --data_dir /home/ujan/Downloads/aishell1 --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-tiny --data_dir /home/ujan/Datasets --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

# change path_remap, batch_size, epochs, csv names between dgx and z8



# 86

rm -r /media/ujan/MHST/dataset

export HF_DATASETS_CACHE="/media/ujan/MHST/dataset"

CUDA_VISIBLE_DEVICES=1,2,3 python asr_seq2seq_no_cache.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-medium  --data_dir /home/ujan/Datasets --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

CUDA_VISIBLE_DEVICES=1,2,3 python asr_seq2seq_with_cache.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-small  --data_dir /home/ujan/Datasets --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end --max_test_samples 100
# --max_train_samples 1000 --max_eval_samples 100 --max_test_samples 100 

CUDA_VISIBLE_DEVICES=1,2,3 python asr_seq2seq_no_cache.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-medium  --data_dir /media/ujan/MHST/far_field --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end

accelerate launch whisper_accelerate.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-small  --data_dir /home/ujan/Datasets 
# --max_train_samples 1000 --max_eval_samples 100 --eval_steps 10

# 76
export HF_DATASETS_CACHE="/raid/cache"
or
export HF_DATASETS_CACHE="/media/ujan/asr/cache"

accelerate launch whisper_accelerate.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/whisper-medium  --data_dir /home/ujan/Datasets --train_batch_size 4 --train_steps 200000 --eval_steps 2000


# cloud
accelerate launch whisper_cloud.py --model_name_or_path openai/whisper-small --data_dir Ujan/asr_testset_zh_16k
accelerate launch whisper_cloud_hi.py --model_name_or_path openai/whisper-small --max_test_samples 300




# eval

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-300m --data_dir /home/ujan/Downloads/aishell1 --do_eval ---predict_with_generate --fp16

accelerate launch whisper_cloud_eval.py --model_name_or_path Ujan/whisper-small-Datasets  --data_dir Ujan/asr_testset_zh_16k 


# quant

python whisper_cloud_quant.py --model_name_or_path Ujan/whisper-small-Datasets  --data_dir Ujan/asr_testset_zh_16k


# eval whisper
python asr_seq2seq.py --model_name_or_path /home/dysl-ai/Documents/ASR/speech-processing/models/checkpoint-12000/ --data_dir /home/dysl-ai/Documents/ASR/speech-processing/src/finetuning/whisper/chinese_movies_dataset --do_eval --predict_with_generate --fp16 --per_device_eval_batch_size 4
