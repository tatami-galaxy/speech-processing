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
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import argparse
from argparse import ArgumentParser
import math

import datasets
import evaluate
import numpy as np
import torch
import datasets
from datasets import Audio
import transformers
from transformers import AdamW
from datasets import DatasetDict, load_dataset
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm

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
    get_scheduler,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.utils.tensorboard import SummaryWriter

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


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
    args,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch[args.text_column])
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
    vocab_set = set()
    for split in datasets:
        vocab_set = vocab_set | set(vocabs[split]["vocab"][0])

    vocab_list = list(vocab_set)
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
        default="mozilla-foundation/common_voice_13_0",
        help="Path to dataset"
    )
    argp.add_argument(
        "--sampling_rate",
        default=16000,
        type=int,
        help="sampling rate",
    )
    argp.add_argument(
        "--data_lang",
        default='hi',
        type=str,
    )
    argp.add_argument(
        '--max_train_samples',
        type=int,
        default=None
    )
    argp.add_argument(
        '--max_test_samples',
        type=int,
        default=None
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
        '--num_workers',
        type=int,
        default=os.cpu_count(),  # None
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
        default=1.0, # 0.0
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
        '--output_dir',
        type=str,
        default=None,
    )

    # model config args
    argp.add_argument(
        '--freeze_feature_encoder',
        action="store_true",
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
        default=0.05, # 0.3?
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
        default=0.05, # 0.1?
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
        '--group_by_length',
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    argp.add_argument(
        '--train_batch_size',
        type=int,
        default=16 #32
    )
    argp.add_argument(
        '--eval_batch_size',
        type=int,
        default=8 #16
    )
    argp.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1
    )
    argp.add_argument(
        '--eval_accumulation_steps',
        type=int,
        default=None
    )
    argp.add_argument(
        "--train_steps",
        type=int,
        default=40000, # None
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    argp.add_argument(
        '--eval_steps',
        type=int,
        default=1000
    )
    argp.add_argument(
        '--warmup_steps',
        type=int,
        default=0,
    )
    argp.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4
    )
    argp.add_argument(
        '--weight_decay',
        type=float,
        default=0.0
    )
    argp.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    argp.add_argument(
        "--adam_beta2",
        type=float,
        default=0.98,
        help="Beta2 for AdamW optimizer",
    )
    argp.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-06,
        help="Epsilon for AdamW optimizer",
    )
    argp.add_argument(
        '--lr_scheduler_type',
        type=str,
        default='linear'
    )    

    # hardware args
    argp.add_argument(
        "--mixed_precision",
        default='no',
        type=str,
    )


    # parse cli arguments
    args = argp.parse_args() 


    # set seed before initializing model.
    set_seed(args.seed)

    # check if data path exists
    if args.data_dir is None:
        raise ValueError(
            f"pass in dataset directory"
        )
    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/wav2vec2/'+model_str+'_'+data_str
    print('output directory set to : {}'.format(args.output_dir))

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.learning_rate,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
    }
    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)


    # Load Datasets and Models #

    # load dataset
    print('loading dataset from {}'.format(args.data_dir))

    dataset= DatasetDict()
    dataset["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation")
    dataset["test"] = load_dataset(args.data_dir, args.data_lang, split="test")

    with accelerator.main_process_first():
        # remove unused columns
        dataset = dataset.remove_columns(
            [
                "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant",
            ]
        )

    # select small dataset for testing
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    if args.max_test_samples is not None:
        dataset["test"] = dataset["test"].select(range(args.max_test_samples))


    # Remove Special Characters #
    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch[args.text_column] = re.sub(chars_to_ignore_regex, "", batch[args.text_column]).lower() + " "
        else:
            batch[args.text_column] = batch[args.text_column].lower() + " "
        return batch

    with accelerator.main_process_first(): #
        dataset = dataset.map(
            remove_special_characters,
            #remove_columns=[args.text_column],
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
    # save vocab in training output dir
    tokenizer_kwargs = {}

    vocab_file = os.path.join(args.output_dir, "vocab.json")

    if os.path.isfile(vocab_file):
        try:
            os.remove(vocab_file)
        except OSError:
            # in shared file-systems it might be the case that
            # two processes try to delete the vocab file at the some time
            pass

    if not os.path.isfile(vocab_file):
        os.makedirs(args.output_dir, exist_ok=True)
        vocab_dict = create_vocabulary_from_data(
            dataset,
            args,
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
        args.output_dir,
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
    with accelerator.main_process_first():
        dataset = dataset.cast_column(
            args.audio_column, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # derive max & min input length for sample rate & max duration
    max_input_length = args.max_duration * feature_extractor.sampling_rate
    min_input_length = args.min_duration * feature_extractor.sampling_rate


    # Preprocessing Datasets #

    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[args.audio_column]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}

        batch["labels"] = tokenizer(batch[args.text_column], **additional_kwargs).input_ids
        return batch

    with accelerator.main_process_first():
        vectorized_dataset = dataset.map(
            prepare_dataset,
            remove_columns=next(iter(dataset.values())).column_names,
            num_proc=args.num_workers,
            desc="preprocess dataset",
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        # filter data that is shorter than min_input_length
        print('filtering')
        vectorized_dataset = vectorized_dataset.filter(
            is_audio_in_length_range,
            num_proc=args.num_workers,
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

    # Activate gradient checkpointing if needed
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Freeze Encoder #
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Define evaluation metrics during training, *i.e.* word error rate, character error rate

    # load from cer metric
    cer_metric = evaluate.load('cer')
    wer_metric = evaluate.load('wer')

    # dataloaders
    train_dataloader = DataLoader(
        vectorized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size
    )
    eval_dataloader = DataLoader(
        vectorized_dataset["test"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size
    )

    # Optimizer
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.learning_rate,
        betas=[args.adam_beta1, args.adam_beta2],
        eps=args.adam_epsilon,
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps,
    )

    # Train
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0
    starting_epoch = 0


    # summarywriter for tensorbaord
    # writer will output to ./runs/ directory by default
    writer = SummaryWriter()


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()

        for step, batch in enumerate(train_dataloader):

            # forward
            outputs = model(**batch)


            # divide loss by gradient accumulation steps since gradients
            # are accumulated for multiple backward passes in PyTorch
            loss = outputs.loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            # update step
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )

                # update parameters
                optimizer.step()
                optimizer.zero_grad()

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                progress_bar.update(1)
                completed_steps += 1


            # eval
            if (step + 1) % (args.gradient_accumulation_steps * args.eval_steps) == 0:
                model.eval()

                # init logs
                val_logs = {
                    "val_loss": 0,
                    "val_cer" : 0
                }

                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        labels = batch["labels"]
                        outputs = model(**batch)

                    val_logs["val_loss"] += outputs.loss

                    pred_ids = torch.argmax(outputs.logits, dim=-1)
                    labels[labels == -100] = processor.tokenizer.pad_token_id
                    pred_str = processor.batch_decode(pred_ids)
                    # we do not want to group tokens when computing the metrics
                    label_str = processor.batch_decode(labels, group_tokens=False)
                    # CER
                    cer_metric.add_batch(predictions=pred_str, references=label_str)

                # compute metric
                val_logs["val_cer"] = cer_metric.compute()



            # log all results
            if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0:
                loss.detach()

                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather_for_metrics(loss).sum()

                train_logs = {
                    "loss": (loss * args.gradient_accumulation_steps),
                    "lr": torch.tensor(optimizer.param_groups[0]["lr"]),
                }
                log_str = ""
                for k, v in train_logs.items():
                    log_str += "| {}: {:.3e}".format(k, v.item())

                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    #if is_wandb_available():
                        #wandb.log(train_logs)

                    # tensorboard logging
                    writer.add_scalar("loss", train_logs["loss"], step+1)
                    writer.add_scalar("lr", train_logs["lr"], step+1)





    # Training Arguments #

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=args.group_by_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        #eval_accumulation_steps=args.eval_accumulation_steps,
        evaluation_strategy=args.evaluation_strategy,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
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
