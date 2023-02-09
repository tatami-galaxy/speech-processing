#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

""" Fine-tuning a 🤗 Transformers CTC model for automatic speech recognition"""

import functools
import json
import os
from os.path import dirname, abspath
import re
from typing import List, Optional
from argparse import ArgumentParser

import datasets
from datasets import DatasetDict, load_dataset

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.27.0.dev0")

#require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )


    # take union of all unique characters in each dataset
    vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["validation"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # remove some special characters from the datasets
    # that make training complicated and do not help in transcribing the speech
    # E.g. characters, such as `,` and `.` do not really have an acoustic characteristic
    # that could be easily picked up by the model
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    argp = ArgumentParser()

    # CLI Arguments #

    argp.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed"
    )

    argp.add_argument(
        '--dataset',
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="The configuration name of the dataset to use (via the datasets library)"
    )

    argp.add_argument(
        '--dataset_config',
        default="zh-CN",
        type=str,
        help="The configuration name of the dataset to use (via the datasets library)"
    )

    argp.add_argument(
        '--train_split',
        type=str,
        default="train",
        help="The name of the training data set split to use (via the datasets library). Defaults to train."
    )

    argp.add_argument(
        '--eval_split',
        type=str,
        default="validation",
        help="The name of the evaluation data set split to use (via the datasets library). Defaults to validation."
    )

    argp.add_argument(
        '--test_split',
        type=str,
        default="test",
        help="The name of the test data set split to use (via the datasets library). Defaults to test."
    )

    argp.add_argument(
        '--audio_column',
        type=str,
        default="audio",
        help="The name of the dataset column containing the audio data. Defaults to audio for cv."
    )

    argp.add_argument(
        '--text_column',
        type=str,
        default="sentence",
        help="The name of the dataset column containing the text data. Defaults to sentence for cv."
    )

    argp.add_argument(
        '--overwrite_output_dir',
        type=bool,
        default=True,
        help="Overwrite output directory or not."
    )

    argp.add_argument(
        '--preprocessing_num_workers',
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing."
    )

    argp.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help="Truncate the number of training examples to this."
    )

    argp.add_argument(
        '--max_eval_samples',
        type=int,
        default=None,
        help="Truncate the number of validation examples to this."
    )

    argp.add_argument(
        '--max_test_samples',
        type=int,
        default=None,
        help="Truncate the number of test examples to this."
    )

    argp.add_argument(
        '--max_duration',
        type=float,
        default=20.0,
        help="Filter audio files that are longer than max_duration."
    )

    argp.add_argument(
        '--min_duration',
        type=float,
        default=0.0,
        help="Filter audio files that are shorter than min_duration."
    )

    argp.add_argument(
        '--unk_token',
        type=str,
        default="[UNK]",
        help="The unk token for the tokenizer."
    )

    argp.add_argument(
        '--pad_token',
        type=str,
        default="[PAD]",
        help="The pad token for the tokenizer."
    )

    argp.add_argument(
        '--word_delimiter_token',
        type=str,
        default="|",
        help="The word delimiter token for the tokenizer."
    )

    argp.add_argument(
        '--model_name_or_path',
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )

    argp.add_argument(
        '--tokenizer_name_or_path',
        type=str,
        default=None,
        help="Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
    )

    argp.add_argument(
        '--output_dir',
        type=str,
        default=root+'/data/processed/cv/',  # take into account tokenizer config maybe?
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )


    # parse cli arguments
    args = argp.parse_args() 

    # set seed before initializing model.
    set_seed(args.seed)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path for tokenizer"
        )

    # check if output directory exist
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # Load the dataset #
    print('Loading dataset {} : {}'.format(args.dataset, args.dataset_config))
    raw_datasets = DatasetDict()

    raw_datasets["train"] = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.train_split,
    )
    raw_datasets["validation"] = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.eval_split,
    )
    raw_datasets["test"] = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.test_split,
    )

    #raw_datasets.cleanup_cache_files()

    if args.audio_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column '{args.audio_column}' not found in dataset '{args.dataset}'."
            " Make sure to set `--audio_column` to the correct audio column - one of"
            f" {', '.join(raw_datasets['train'].column_names)}."
        )

    if args.text_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--text_column {args.text_column} not found in dataset '{args.dataset}'. "
            "Make sure to set `--text_column` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

    if args.max_eval_samples is not None:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(args.max_eval_samples))

    if args.max_test_samples is not None:
        raw_datasets["test"] = raw_datasets["test"].select(range(args.max_test_samples))


    text_column = args.text_column

    # Remove Special Characters #

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column]).lower() + " "
        else:
            batch["target_text"] = batch[text_column].lower() + " "
        return batch

    raw_datasets = raw_datasets.map(
        remove_special_characters,
        remove_columns=[text_column],
        desc="remove special characters from datasets",
    )

    # save special tokens for tokenizer
    word_delimiter_token = args.word_delimiter_token
    unk_token = args.unk_token
    pad_token = args.pad_token

    # load the config as we might need it to create the tokenizer
    # load config
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    # Make Vocab #

    # if no tokenizer file is defined,
    # we create the vocabulary of the model by extracting all unique characters from
    # the training and evaluation datasets
    # we need to make sure that only first rank saves vocabulary
    # make sure all processes wait until vocab is created
    tokenizer_name_or_path = args.tokenizer_name_or_path
    tokenizer_kwargs = {}
    if tokenizer_name_or_path is None:
        # save vocab in training output dir
        tokenizer_name_or_path = args.output_dir

        vocab_file = os.path.join(tokenizer_name_or_path, "vocab.json")

        if args.overwrite_output_dir and os.path.isfile(vocab_file):
            try:
                os.remove(vocab_file)
            except OSError:
                # in shared file-systems it might be the case that
                # two processes try to delete the vocab file at the some time
                pass

        if not os.path.isfile(vocab_file):
            os.makedirs(tokenizer_name_or_path, exist_ok=True)
            vocab_dict = create_vocabulary_from_data(
                raw_datasets,
                word_delimiter_token=word_delimiter_token,
                unk_token=unk_token,
                pad_token=pad_token,
            )

            # save vocab dict to be loaded into tokenizer
            with open(vocab_file, "w") as file:
                json.dump(vocab_dict, file)

        # if tokenizer has just been created
        # it is defined by `tokenizer_class` if present in config else by `model_type`
        tokenizer_kwargs = {
            "config": config if config.tokenizer_class is not None else None,
            "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
            "unk_token": unk_token,
            "pad_token": pad_token,
            "word_delimiter_token": word_delimiter_token,
        }

    # instantiate the feature extractor, tokenizer and model
    # for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.

    # Tokenizer and Feature extractor #

    # load feature_extractor and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_kwargs,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name_or_path)

    # preprocess the datasets including loading the audio, resampling and normalization
    # `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`

    # make sure that dataset decodes audio with correct sampling rate
    dataset_sampling_rate = next(iter(raw_datasets.values())).features[args.audio_column].sampling_rate
    if dataset_sampling_rate != feature_extractor.sampling_rate:
        raw_datasets = raw_datasets.cast_column(
            args.audio_column, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = args.max_duration * feature_extractor.sampling_rate
    min_input_length = args.min_duration * feature_extractor.sampling_rate
    audio_column = args.audio_column
    num_workers = args.preprocessing_num_workers


    # Preprocessing Datasets #

    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=num_workers,
        desc="preprocess dataset",
    )

    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    # filter data that is shorter than min_input_length
    print('filtering')
    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    print('Saving..')
    vectorized_datasets.save_to_disk(args.output_dir+'/vectorized_dataset')
    # save feature extractor, tokenizer and config
    tokenizer.save_pretrained(args.output_dir+'/tokenizer')
    feature_extractor.save_pretrained(args.output_dir+'/feature_extractor')
    config.save_pretrained(args.output_dir+'/config')

    # write model name
    with open(args.output_dir+'model_name.txt', 'w') as f:
        f.write(args.model_name_or_path)

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with ``args.preprocessing_only`` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step ``args.preprocessing_only`` can then be set to `False` to load the
    # cached dataset
    #print("Data preprocessing finished. Files cached at {}".format(vectorized_datasets.cache_files))
    print("Data preprocessing finished.")
    print("Stored at {}".format(args.output_dir))

    return


if __name__ == "__main__":
    main()
