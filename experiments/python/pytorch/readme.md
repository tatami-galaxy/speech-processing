# runpod

accelerate launch whisper_cv_accelerate.py 
# --model_name_or_path openai/whisper-small gradient_accumulation_steps 4 --resume_from_checkpoint "checkpoint" --skip_steps --max_train_samples 1000 --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --atrain_batch_size 4 --eval_batch_size 4 --preprocessing_num_workers 1

accelerate launch whisper_cv_distil.py 
# --model_name_or_path openai/whisper-tiny --teacher_name_or_path "teacher" gradient_accumulation_steps 4 --resume_from_checkpoint "checkpoint" --skip_steps --max_train_samples 1000 --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --preprocessing_num_workers 1



# eval

python whisper_cv_eval.py --model_name_or_path openai/whisper-small --checkpoint "checkpoint path" --predict_with_generate --fp16
# --max_test_samples 100 --save_steps 100 --eval_steps 100 --logging_steps 100 --per_device_eval_batch_size 4 --eval_accumulation_steps 4 --preprocessing_num_workers 1


# trace
python whisper_trace.py --model_name_or_path Ujan/whisper-small-Datasets --data_dir Ujan/asr_testset_zh_16k --max_test_samples 100
python whisper_loop_gen.py --model_name_or_path Ujan/whisper-small-Datasets --data_dir Ujan/asr_testset_zh_16k --max_test_samples 100

python whisper_loop_gen.py --model_name_or_path openai/whisper-small --data_dir mozilla-foundation/common_voice_11_0 --cv
python whisper_loop_gen.py --model_name_or_path Ujan/whisper-small-Datasets --data_dir mozilla-foundation/common_voice_11_0 --cv

