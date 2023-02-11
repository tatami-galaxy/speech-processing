from os.path import dirname, abspath
from transformers import AutoModelForCTC

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)

model_name = 'facebook/wav2vec2-xls-r-300m'
output_dir = root+'/models/pretrained_models/'+(model_name.split('/')[-1])

model = AutoModelForCTC.from_pretrained(model_name)
model.save_pretrained(output_dir)