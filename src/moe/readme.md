accelerate launch whisper_cloud_hi.py --activation relu --max_test_samples 100

# sparse act
accelerate launch whisper_sparse_act.py --activation relu --train_batch_size 1 --eval_batch_size 1 --max_train_samples 50 --max_test_samples 50 --model_name_or_path /users/ujan/speech-processing/models/whisper/moe/whisper-small-cv-moe/checkpoint-6000