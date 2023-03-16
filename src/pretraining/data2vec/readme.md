accelerate launch data2vec_pretraining.py --model_name_or_path /home/ujan/speech-processing/models/pretrained_models/data2vec-en-base --data_dir /home/ujan/Datasets --preprocessing_num_workers 4 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --max_train_steps 1000 --max_train_samples 1000 --max_eval_samples 100 --saving_steps 100 --logging_steps 100 --num_warmup_steps 0 --gradient_checkpointing

change path_remap, batch_size, epochs, csv names between dgx and z8

