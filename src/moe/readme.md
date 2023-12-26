accelerate launch whisper_cloud_hi.py --activation relu --max_test_samples 100

# sparse act
accelerate launch check_activation.py --activation relu --max_test_samples 50 --model_name_or_path /home/ujan/speech-processing/models/whisper/whisper-small_common_voice_13_0_relu/checkpoint-6000

# co-act
accelerate launch calc_co-activation.py --activation relu --max_test_samples 50 --model_name_or_path /home/ujan/speech-processing/models/whisper/whisper-small_common_voice_13_0_relu/checkpoint-6000 --low_ram