# runpod

python whisper_cv.py --model_name_or_path openai/whisper-small --do_train --do_eval --overwrite_output_dir --gradient_checkpointing --predict_with_generate --fp16 --load_best_model_at_end --overwrite_output_dir
# --max_train_samples 1000 --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --per_device_train_batch_size 4 --per_device_train_batch_size 4 --eval_accumulation_steps 4



# eval

python whisper_cv_eval.py --model_name_or_path openai/whisper-small --checkpoint "checkpoint path" --predict_with_generate --fp16 --load_best_model_at_end 
# --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --per_device_eval_batch_size 4 --eval_accumulation_steps 4

