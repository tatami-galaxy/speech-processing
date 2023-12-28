"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

import os
from os.path import dirname, abspath
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict, Audio

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import GenerationConfig
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
#from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers import WhisperConfig
from modeling_whisper_moe import WhisperForConditionalGeneration

from torch.utils.data.dataloader import DataLoader
from transformers import set_seed
import argparse
from accelerate import Accelerator

from k_means_constrained import KMeansConstrained
import numpy as np

#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=50000))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)
    

def cluster(model, k_means, encoder=True):

    config = model.config

    if encoder:
        num_layers = config.encoder_layers
        ffn_dim = config.encoder_ffn_dim
    else:
        num_layers = config.decoder_layers
        ffn_dim = config.decoder_ffn_dim

    layer_bar = tqdm(range(num_layers))
    for l in range(config.encoder_layers):
        if encoder:
            W1 = model.model.encoder.layers[l].fc1.weight
            k_means.fit_predict(W1.detach())
            labels = k_means.labels_
            quit()
        else:
            pass



def run():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default="openai/whisper-small",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--activation",
        default='relu',
        type=str,
        help="change model activation function",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--model_lang",
        default='hindi',
        type=str,
    )
    parser.add_argument(
        "--task",
        default='transcribe',
        type=str,
    )
    parser.add_argument(
        '--forced_decoder_ids',
        type=List[List[int]],
        default=None,
        help="""A list of pairs of integers which indicates a mapping from generation indices to token indices
                that will be forced before sampling. For example, [[0, 123]] means the first generated token
                will always be a token of index 123."""
    )
    parser.add_argument(
        '--suppress_tokens',
        type=List[int],
        default=None,
        help="A list of tokens that will be suppressed at generation."
    )
    parser.add_argument(
        "--mixed_precision",
        default='fp16',
        type=str,
    )
    parser.add_argument(
        "--num_experts",
        default=96,
        type=str,
    )

    # parse args
    args = parser.parse_args()
    args.eval_batch_size = 1

    # set seed
    set_seed(args.seed)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )
    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    config = WhisperConfig.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # working. detect sparse activation. try changing hardcoded gelus to relu
    if args.activation is not None:
        config.activation_function = args.activation
        print('activation changed to {}'.format(config.activation_function))

    # model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        config=config,
        #cache_dir=args.cache_dir if args.cache_dir else None,
    )
    #model.config.forced_decoder_ids = None
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    model.config.suppress_tokens = []

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False


    # prepare everything for accelerator
    model = accelerator.prepare(model)

    expert_size = model.config.encoder_ffn_dim / args.num_experts 
    k_means = KMeansConstrained(
        n_clusters=args.num_experts,
        size_min=expert_size,
        size_max=expert_size,
        random_state=0
    )
    cluster(model, k_means, encoder=True)
    cluster(model, k_means, encoder=False)

    #print(model.model.encoder.layers[3].fc1.weight.shape)
    #print(model.model.encoder.layers[3].fc1.bias.shape)


    X = torch.tensor([[1, 2], [1, 4], [1, 0],
                  [4, 2], [4, 4], [4, 0]])
    clf = KMeansConstrained(
        n_clusters=2,
        size_min=2,
        size_max=5,
        random_state=0
    )
    # fit_predict(X, y=None)
    # X : {array-like, sparse matrix}, shape = [n_samples, n_features]
    clf.fit_predict(X)
    accelerator.print(clf.cluster_centers_)
    accelerator.print(clf.labels_)

      


if __name__ == "__main__":

    run()


