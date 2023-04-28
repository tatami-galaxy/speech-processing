# runpod

python whisper_cv.py --model_name_or_path openai/whisper-small --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end
# --max_train_samples 1000 --max_eval_samples 100 --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 '--per_device_train_batch_size 8 --eval_accumulation_steps 4



# eval

python asr_seq2seq.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/wav2vec2-xls-r-300m --data_dir /home/ujan/Downloads/aishell1 --do_eval ---predict_with_generate --fp16

