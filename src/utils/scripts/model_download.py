from os.path import dirname, abspath
from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoTokenizer, AutoModelForSpeechSeq2Seq
import argparse
from argparse import ArgumentParser

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)

argp = ArgumentParser()

# CLI Arguments #


argp.add_argument(
    '--model_name',
    type=str,
    default=None,
    help="Path to pretrained model or model identifier from huggingface.co/models"
)


# parse input arguments
# parse cli arguments
args = argp.parse_args() 


output_dir = root+'/models/pretrained_models/'+(args.model_name.split('/')[-1])

# model
#model = AutoModelForCTC.from_pretrained(args.model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name)
model.save_pretrained(output_dir)

# feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
feature_extractor.save_pretrained(output_dir)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.save_pretrained(output_dir)

print('done.')