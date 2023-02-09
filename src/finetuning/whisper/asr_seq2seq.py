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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import argparse
from argparse import ArgumentParser

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
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
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    do_lower_case: bool = field(
        default=True,
        metadata={"help": "Whether the target text should be lower cased."},
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )


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
    # parse input arguments

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
        '--processed_data_dir',
        type=str,
        default=root+'/data/processed/cv/',
        help="Path to processed data, tokenizer, extractor."
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
        default=root+'/models/whisper/',
        help="Where do you want to store the pretrained models downloaded from huggingface.co"
    )


    # model config args
    argp.add_argument(
        '--language',
        type=str,
        default=None,
        help="Language for multilingual fine-tuning."
    )
    argp.add_argument(
        '--task',
        type=str,
        default='transcribe',
        help="Task, either `transcribe` for speech recognition or `translate` for speech translation."
    )
    argp.add_argument(
        '--freeze_feature_encoder',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to freeze the feature encoder layers of the model."
    )
    argp.add_argument(
        '--freeze_encoder',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to freeze the entire encoder of the seq2seq model."
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
        '--predict_with_generate',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to calculate generative metrics"
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

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = Seq2SeqTrainingArguments.get_process_log_level() ####
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, n_gpu: {args.n_gpu}"
        f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", args)

    # detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.seed)

    # Load Pre-processed Datasets and Models #

    # load dataset
    print('loading processed data from {}'.format(args.processed_data_dir))
    with open(args.processed_data_dir+'model_name.txt') as f:
        processed_config = f.readline()
        print('dataset was processed with config from {}'.format(processed_config))
    vectorized_datasets = load_from_disk(args.processed_data_dir+'/vectorized_dataset')

    if args.max_train_samples is not None:
        vectorized_datasets["train"] = vectorized_datasets["train"].select(range(args.max_train_samples))

    if args.max_eval_samples is not None:
        vectorized_datasets["validation"] = vectorized_datasets["validation"].select(range(args.max_eval_samples))

    if args.max_test_samples is not None:
        vectorized_datasets["test"] = vectorized_datasets["test"].select(range(args.max_test_samples))

    # load config
    config = AutoConfig.from_pretrained(args.processed_data_dir+'/config')

    # load tokenizer
    print('loading tokenizer..')
    tokenizer = AutoTokenizer.from_pretrained(args.processed_data_dir+'/tokenizer')

    # load feature exatractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.processed_data_dir+'/feature_extractor')
    config.update({"forced_decoder_ids": args.forced_decoder_ids, "suppress_tokens": args.suppress_tokens})

     # load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name_or_path, config=config)  # whisper?


    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if args.language is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=args.language, task=args.task)

    # resample speech dataset if necessary
    #dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    #if dataset_sampling_rate != feature_extractor.sampling_rate:
        #raw_datasets = raw_datasets.cast_column(
            #data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        #)


    # Load Metric
    # load offline
    metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

        cer = metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    # create a single speech processor
    if is_main_process(args.local_rank):
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    processor = AutoProcessor.from_pretrained(args.output_dir)

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
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
        metric_for_best_model=args.metric_for_best_model
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if args.do_eval else None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(vectorized_datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation #

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        metrics["eval_samples"] = len(vectorized_datasets["eval"])

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
