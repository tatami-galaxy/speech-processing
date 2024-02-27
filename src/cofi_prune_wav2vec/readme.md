export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

### Cloud

accelerate launch wav2vec2.py --model_name_or_path facebook/wav2vec2-large-xlsr-53 --max_test_samples 300 --ctc_loss_reduction none --ctc_zero_infinity
