#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.
from functools import partial
import json
import logging
import os
from os.path import dirname, abspath
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import argparse
from argparse import ArgumentParser

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

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
        default="mozilla-foundation/common_voice_11_0",
        help="Path to dataset or dataset string"
    )
    argp.add_argument(
        '--data_lang',
        type=str,
        default="hi",  # zh-CN
        help="Path to dataset or dataset string"
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
        '--audio_column',
        type=str,
        default="audio",
        help="The name of the dataset column containing the audio data. Defaults to audio for cv."
    )

    argp.add_argument(
        '--text_column',
        type=str,
        default="sentence", # sentence, transcript
        help="The name of the dataset column containing the text data. Defaults to sentence for cv."
    )
    argp.add_argument(
        '--preprocessing_num_workers',
        type=int,
        default=8, # None
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
        '--preprocessing_only',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to only do data preprocessing and skip training."
    )



    # model args
    argp.add_argument(
        '--model_name_or_path',
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    argp.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="Path to model to eval"
    )
    argp.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )


    # model config args
    argp.add_argument(
        '--model_lang',
        type=str,
        default="Hindi"
    )
    argp.add_argument(
        '--task',
        type=str,
        default="transcribe"
    )
    argp.add_argument(
        '--forced_decoder_ids',
        type=List[List[int]],
        default=None,
        help="""A list of pairs of integers which indicates a mapping from generation indices to token indices 
                that will be forced before sampling. For example, [[0, 123]] means the first generated token 
                will always be a token of index 123."""
    )
    argp.add_argument(
        '--suppress_tokens',
        type=List[int],
        default=None,
        help="A list of tokens that will be suppressed at generation."
    )
    argp.add_argument(
        '--freeze_encoder',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to freeze the transformer encoder of the model."
    )



    # model training args
    argp.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=8 #16
    )
    argp.add_argument(
        '--eval_accumulation_steps',
        type=int,
        default=None
    )
    argp.add_argument(
        '--evaluation_strategy',
        type=str,
        default="steps"
    )
    argp.add_argument(
        '--predict_with_generate',
        default=False,
        action=argparse.BooleanOptionalAction
    )
    argp.add_argument(
        '--generation_max_length',
        type=int,
        default=225
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
    


    # hardware args
    argp.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help="Rank of the process during distributed training."
    )
    #argp.add_argument(
        #'--n_gpu',
        #type=int,
        #default=1,
        #help="Number of GPUs to use."
    #)
    argp.add_argument(
        '--fp16',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."
    )


    # parse input arguments
    # parse cli arguments
    args = argp.parse_args() 


    # check if data path exists
    if args.data_dir is None:
        raise ValueError(
            f"pass in dataset directory or dataset string"
        )
    #args.processed_data_dir = root+'/data/processed/'+args.processed_data_dir+'/'
    #if not os.path.isdir(args.data_dir):
        #raise ValueError(
            #f"data directory does not exist"
        #)


    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str
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
    send_example_telemetry("run_speech_recognition_seq2seq", args)

    # seq2seq training args

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        #group_by_length=args.group_by_length,
        #per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        #gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        #evaluation_strategy=args.evaluation_strategy,
        #num_train_epochs=args.num_train_epochs,
        #max_steps=args.max_steps,
        fp16=args.fp16,
        #gradient_checkpointing=args.gradient_checkpointing,
        #save_steps=args.save_steps,
        #eval_steps=args.eval_steps,
        #logging_steps=args.logging_steps,
        #learning_rate=args.learning_rate,
        #lr_scheduler_type=args.lr_scheduler_type,
        #weight_decay=args.weight_decay,
        #warmup_steps=args.warmup_steps,
        #save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate, ##
        generation_max_length=args.generation_max_length,
        #load_best_model_at_end=args.load_best_model_at_end,
        #metric_for_best_model=args.metric_for_best_model,  # cer
        greater_is_better=False, 
        report_to='tensorboard'
    )



    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    

    # load dataset

    ## use hindi or smaller dataset ##
    common_voice = DatasetDict()
    #common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test", use_auth_token=True)

    # remove columns
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])


    #raw_datasets.cleanup_cache_files()

    # check audio column, text column names
    if args.audio_column not in common_voice["test"].column_names:
        raise ValueError(
            f"--audio_column '{args.audio_column}' not found in dataset '{args.data_dir}'."
            " Make sure to set `--audio_column` to the correct audio column - one of"
            f" {', '.join(common_voice['test'].column_names)}."
        )

    if args.text_column not in common_voice["test"].column_names:
        raise ValueError(
            f"--text_column {args.text_column} not found in dataset '{args.data_dir}'. "
            "Make sure to set `--text_column` to the correct text column - one of "
            f"{', '.join(common_voice['test'].column_names)}."
        )


    if args.max_test_samples is not None:
        common_voice["test"] = common_voice["test"].select(range(args.max_test_samples))




    # Load pretrained model, tokenizer, and feature extractor
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )

    config.update({"forced_decoder_ids": args.forced_decoder_ids, "suppress_tokens": args.suppress_tokens})

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir,
        #use_fast=model_args.use_fast_tokenizer,
        #revision=model_args.model_revision,
        #use_auth_token=True if model_args.use_auth_token else None,
    )
    # load from checkpoint for eval
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        #args.model_name_or_path,
        args.checkpoint,
        config=config,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    if args.freeze_encoder:
        model.freeze_encoder()
        #model.model.encoder.gradient_checkpointing = False

    if args.model_lang is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)


    # resample speech dataset if necessary
    #dataset_sampling_rate = next(iter(raw_datasets.values())).features[args.audio_column].sampling_rate
    #if dataset_sampling_rate != feature_extractor.sampling_rate:
    common_voice = common_voice.cast_column(
        "audio", datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = args.max_duration * feature_extractor.sampling_rate
    min_input_length = args.min_duration * feature_extractor.sampling_rate
    audio_column_name = args.audio_column
    num_workers = args.preprocessing_num_workers
    text_column_name = args.text_column
    model_input_name = feature_extractor.model_input_names[0]
    #do_lower_case = args.do_lower_case



    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # process targets
        #input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    #with training_args.main_process_first(desc="dataset map pre-processing"):
    vectorized_datasets = common_voice.map(
        prepare_dataset,
        remove_columns=next(iter(common_voice.values())).column_names,
        num_proc=args.preprocessing_num_workers,
        keep_in_memory=True, # no cache
        desc="preprocess dataset",
    )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if args.preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return


    # load metric
    metric = evaluate.load("cer")
    # load offline
    #metric = evaluate.load("/home/ujan/Downloads/evaluate/metrics/cer/cer.py")


    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        cer = metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}


    # create a single speech processor
    if is_main_process(training_args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)


    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        language=args.model_lang,
        task=args.task,
    )


    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=vectorized_datasets["test"],
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
    )


    # Evaluation

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(
        metric_key_prefix="eval",
        max_length=args.generation_max_length,
        #num_beams=training_args.generation_num_beams,
    )
    max_eval_samples = (
        args.max_eval_samples if args.max_eval_samples is not None else len(vectorized_datasets["test"])
    )
    metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["test"]))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



if __name__ == "__main__":

    main()
