export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# check sparse act
accelerate launch check_activation.py --activation relu --max_test_samples 50 --model_name_or_path /home/ujan/speech-processing/models/whisper/whisper-small_common_voice_13_0_relu/checkpoint-6000

# co-act
accelerate launch co-activation_split.py --activation relu --max_test_samples 50 --model_name_or_path /home/ujan/speech-processing/models/whisper/whisper-small_common_voice_13_0_relu/checkpoint-6000 --low_ram

# balanced k-means
python balanced_k-means_split.py --activation relu --model_name_or_path /home/ujan/speech-processing/models/whisper/whisper-small_common_voice_13_0_relu/checkpoint-6000 

# train moe
accelerate launch whisper_cloud_hi_moe.py --model_name_or_path Ujan/whisper-small_moe_k_means --activation relu --n_experts 96 --max_test_samples 300 --local