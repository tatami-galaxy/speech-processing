import os
import subprocess

os.mkdir('data')
os.mkdir('data/raw')
os.mkdir('data/interim')
os.mkdir('data/processed')

os.mkdir('models')
os.mkdir('models/whisper')
os.mkdir('models/wav2vec2')
os.mkdir('models/data2vec')
os.mkdir('models/pretrained_models')
os.mkdir('models/pretrained_models/cont_pre_training')
 
# Using system() method to
# execute shell commands
subprocess.Popen('huggingface-cli login"', shell=True)