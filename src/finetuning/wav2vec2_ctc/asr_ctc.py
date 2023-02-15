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
from functools import partial
import json
import logging
import os
from os.path import dirname, abspath
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import argparse
from argparse import ArgumentParser

import datasets
import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    Wav2Vec2Processor,
    TrainingArguments,
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


logger = logging.getLogger(__name__)

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


def path_remap(x, args):

    # get audio path
    path_list = x['audio'].split('/')

    for i in range(len(path_list)):
        if path_list[i] == 'wav': break

    new_path = '/'.join(path_list[i:])
    new_path = args.data_dir+'/'+new_path
    x['audio'] = new_path

    return x



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch



def main():

    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

    argp = ArgumentParser()

    # CLI Arguments #


    # seed
    argp.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed"
    )


    # dataset args
    argp.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help="Path to dataset"
    )
    argp.add_argument(
        '--max_train_samples',
        type=int,
        default=None
    )
    argp.add_argument(
        '--max_eval_samples',
        type=int,
        default=None
    )
    argp.add_argument(
        '--max_test_samples',
        type=int,
        default=None
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
        default="transcript",
        help="The name of the dataset column containing the text data. Defaults to sentence for cv."
    )
    argp.add_argument(
        '--preprocessing_num_workers',
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing."
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



    # model args
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
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )


    # model config args
    argp.add_argument(
        '--freeze_feature_encoder',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to freeze the feature encoder layers of the model."
    )
    argp.add_argument(
        '--attention_dropout',
        type=float,
        default=0.0,
        help="The dropout ratio for the attention probabilities."
    )
    argp.add_argument(
        '--activation_dropout',
        type=float,
        default=0.0,
        help="The dropout ratio for activations inside the fully connected layer."
    )
    argp.add_argument(
        '--feat_proj_dropout',
        type=float,
        default=0.0,
        help="The dropout ratio for the projected features."
    )
    argp.add_argument(
        '--hidden_dropout',
        type=float,
        default=0.0,
        help="The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
    )
    argp.add_argument(
        '--final_dropout',
        type=float,
        default=0.0,
        help="The dropout probability for the final projection layer."
    )
    argp.add_argument(
        '--mask_time_prob',
        type=float,
        default=0.05,
        help="""Probability of each feature vector along the time axis to be chosen as the start of the vector 
        span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature 
        vectors will be masked along the time axis."""
    )
    argp.add_argument(
        '--mask_time_length',
        type=int,
        default=10,
        help="Length of vector span to mask along the time axis."
    )
    argp.add_argument(
        '--mask_feature_prob',
        type=float,
        default=0.05,
        help="""Probability of each feature vector along the feature axis to be chosen as the start of the vectorspan 
        to be masked. Approximately ``mask_feature_prob * sequence_length // mask_feature_length`` feature 
        bins will be masked along the time axis."""
    )
    argp.add_argument(
        '--mask_feature_length',
        type=int,
        default=10,
        help="Length of vector span to mask along the feature axis."
    )
    argp.add_argument(
        '--layerdrop',
        type=float,
        default=0.0,
        help="The LayerDrop probability."
    )
    argp.add_argument(
        '--ctc_loss_reduction',
        type=str,
        default="mean",
        help="The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."
    )


    # model training args
    argp.add_argument(
        '--do_train',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to train the model."
    )
    argp.add_argument(
        '--do_eval',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to evaluatte the model."
    )
    argp.add_argument(
        '--overwrite_output_dir',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to overwrite output directory. Need to be False to load checkpoint"
    )
    argp.add_argument(
        '--gradient_checkpointing',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass."
    )
    argp.add_argument(
        '--group_by_length',
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    argp.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16
    )
    argp.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=8
    )
    argp.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1
    )
    argp.add_argument(
        '--eval_accumulation_steps',
        type=int,
        default=16
    )
    argp.add_argument(
        '--evaluation_strategy',
        type=str,
        default="steps"
    )
    argp.add_argument(
        '--num_train_epochs',
        type=int,
        default=30
    )
    argp.add_argument(
        '--save_steps',
        type=int,
        default=1000
    )
    argp.add_argument(
        '--eval_steps',
        type=int,
        default=1000
    )
    argp.add_argument(
        '--logging_steps',
        type=int,
        default=1000
    )
    argp.add_argument(
        '--warmup_steps',
        type=int,
        default=0
    )
    argp.add_argument(
        '--learning_rate',
        type=float,
        default=0.0002
    )
    argp.add_argument(
        '--weight_decay',
        type=float,
        default=0.0
    )
    argp.add_argument(
        '--lr_scheduler_type',
        type=str,
        default='linear'
    )
    argp.add_argument(
        '--save_total_limit',
        type=int,
        default=2
    )
    argp.add_argument(
        '--load_best_model_at_end',
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    argp.add_argument(
        '--metric_for_best_model',
        type=str,
        default='cer'
    )
    

    # model eval args
    argp.add_argument(
        '--eval_metrics',
        type=List[str],
        default=["cer"],
        help="A list of metrics the model should be evaluated on."
    )


    # hardware args
    argp.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help="Rank of the process during distributed training."
    )
    argp.add_argument(
        '--n_gpu',
        type=int,
        default=1,
        help="Number of GPUs to use."
    )
    argp.add_argument(
        '--fp16',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."
    )


    # parse cli arguments
    args = argp.parse_args() 


    # set seed before initializing model.
    set_seed(args.seed)

    # check if processed data path exists
    if args.data_dir is None:
        raise ValueError(
            f"pass in dataset directory"
        )
    #args.processed_data_dir = root+'/data/processed/'+args.processed_data_dir+'/'
    if not os.path.isdir(args.data_dir):
        raise ValueError(
            f"data directory does not exist"
        )

    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/wav2vec2/'+model_str+'_'+data_str
    print('output directory set to : {}'.format(args.output_dir))
    #if not os.path.isdir(args.output_dir):
        #os.mkdir(args.output_dir)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_ctc", args)

    # Detecting last checkpoint.
    # where to store checkpoint? -> output_dir
    # check when checkpoint is loaded
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        print('Checkpoint found')
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, n_gpu: {args.n_gpu} "
        f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", args)



    # Load Datasets and Models #

    # load dataset
    print('loading dataset from {}'.format(args.data_dir))

    # data files
    data_files = {
        'train': args.data_dir+'/train.csv',
        'validation': args.data_dir+'/validation.csv',
        'test': args.data_dir+'/test.csv',
        }

    raw_datasets = load_dataset('csv', data_files=data_files)

    # map to new audio path
    raw_datasets = raw_datasets.map(partial(path_remap, args=args), batched=False)


    #raw_datasets.cleanup_cache_files()

    # check audio column, text column names
    if args.audio_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column '{args.audio_column}' not found in dataset '{args.data_dir}'."
            " Make sure to set `--audio_column` to the correct audio column - one of"
            f" {', '.join(raw_datasets['train'].column_names)}."
        )

    if args.text_column not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--text_column {args.text_column} not found in dataset '{args.data_dir}'. "
            "Make sure to set `--text_column` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )
    text_column = args.text_column

    if args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

    if args.max_eval_samples is not None:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(args.max_eval_samples))

    if args.max_test_samples is not None:
        raw_datasets["test"] = raw_datasets["test"].select(range(args.max_test_samples))


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

    #dataset_sampling_rate = next(iter(raw_datasets.values())).features[args.audio_column].sampling_rate
    #if dataset_sampling_rate != feature_extractor.sampling_rate:
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


    # adapt config
    config.update(
        {
            "feat_proj_dropout": args.feat_proj_dropout,
            "attention_dropout": args.attention_dropout,
            "hidden_dropout": args.hidden_dropout,
            "final_dropout": args.final_dropout,
            "mask_time_prob": args.mask_time_prob,
            "mask_time_length": args.mask_time_length,
            "mask_feature_prob": args.mask_feature_prob,
            "mask_feature_length": args.mask_feature_length,
            "gradient_checkpointing": args.gradient_checkpointing,
            "layerdrop": args.layerdrop,
            "ctc_loss_reduction": args.ctc_loss_reduction,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),  # custom vocab
            "activation_dropout": args.activation_dropout,
        }
    )


    # load model
    model = AutoModelForCTC.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True)

    # Freeze Encoder #
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()


    # Evaluation Metric #

    # instantiate a data collator and the trainer
    # Define evaluation metrics during training, *i.e.* word error rate, character error rate

    # load from file for offline
    eval_metrics = {metric: evaluate.load(metric) for metric in args.eval_metrics}

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        metrics = {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

        return metrics

    # processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Training Arguments #

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=args.group_by_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        report_to='tensorboard'
    )


    # initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if args.do_train else None,
        eval_dataset=vectorized_datasets["validation"] if args.do_eval else None,
        tokenizer=feature_extractor,
    )

    # Start training #

    # Training
    if args.do_train:
        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        #elif os.path.isdir(args.model_name_or_path):
            #checkpoint = args.model_name_or_path
        else:
            checkpoint = None

        print('checkpoint : {}'.format(checkpoint))

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        #max_train_samples = (
            #args.max_train_samples
            #if args.max_train_samples is not None
            #else len(vectorized_datasets["train"])
        #)
        #metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))
        metrics["train_samples"] = len(vectorized_datasets["train"])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation # 

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        #max_eval_samples = (
            #args.max_eval_samples if args.max_eval_samples is not None else len(vectorized_datasets["validation"])
        #)
        #metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["validation"]))
        metrics["eval_samples"] = len(vectorized_datasets["validation"])

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print("Saved in {}".format(args.output_dir))


if __name__ == "__main__":
    main()
