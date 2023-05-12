"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

from functools import partial
import os, re
from os.path import dirname, abspath
import numpy as np
from tqdm.auto import tqdm, trange
from datasets import load_dataset
import transformers, datasets
from transformers import AutoConfig
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import AutoProcessor
import torch
from torch import nn
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import AutoModelForSpeechSeq2Seq
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed
import argparse
from accelerate import Accelerator

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)


def path_remap(x, args):

    # get audio path
    #path_list = x['audio'].split('/')
    path = x['audio']

    #for i in range(len(path_list)):
        #if path_list[i] == 'wav': break

    #new_path = '/'.join(path_list[i:])
    #new_path = args.data_dir+'/'+new_path
    new_path = args.data_dir+'/'+path
    x['audio'] = new_path

    return x


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

    


def train(args, accelerator):

    # load dataset
    print('loading dataset from {}'.format(args.data_dir))

    # data files
    data_files = {
        'train': args.data_dir+'/final_train.csv', # final_train.csv
        'validation': args.data_dir+'/final_dev_short.csv', # final_train.csv
        'test': args.data_dir+'/final_test_short.csv', # final_test.csv
        }

    raw_datasets = load_dataset('csv', data_files=data_files)

    # map to new audio path
    with accelerator.main_process_first():
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

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

        if args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(args.max_eval_samples))

        if args.max_test_samples is not None:
            raw_datasets["test"] = raw_datasets["test"].select(range(args.max_test_samples))

        # resample speech dataset if necessary
        raw_datasets = raw_datasets.cast_column(
            args.audio_column, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # remove punctuations
    def remove_special_characters(batch):
        batch[args.text_column] = re.sub(chars_to_ignore_regex, "", batch[args.text_column]).lower() + " "
        return batch
    
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            desc="remove special characters from datasets",
        )


    # model, tokenizer

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )
    teacher_config = AutoConfig.from_pretrained(
        args.teacher_name_or_path,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )

    model_config.update({"forced_decoder_ids": args.forced_decoder_ids, "suppress_tokens": args.suppress_tokens})
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task="transcribe")

    teacher_config.update({"forced_decoder_ids": args.forced_decoder_ids, "suppress_tokens": args.suppress_tokens})
    teacher.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task="transcribe")

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
    if args.model_lang is not None:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        tokenizer.set_prefix_tokens(language=args.language, task=args.task)


    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )
    teacher = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.teacher_name_or_path,
        config=teacher_config,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if teacher.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False



    # resample speech dataset if necessary
    #dataset_sampling_rate = next(iter(raw_datasets.values())).features[args.audio_column].sampling_rate
    #if dataset_sampling_rate != feature_extractor.sampling_rate:
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.cast_column(
            args.audio_column, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
        )

    # preprocess dataset
    max_input_length = args.max_duration * feature_extractor.sampling_rate
    min_input_length = args.min_duration * feature_extractor.sampling_rate
    audio_column_name = args.audio_column
    num_workers = args.preprocessing_num_workers
    text_column_name = args.text_column
    model_input_name = feature_extractor.model_input_names[0]


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
    
    with accelerator.main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=args.preprocessing_num_workers,
            keep_in_memory=True,  # no cache
            desc="preprocess train dataset",
        )

    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    with accelerator.main_process_first():
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
            keep_in_memory=True
        )


    # cer metric
    #metric = evaluate.load("cer")
    metric = evaluate.load("/home/ujan/Downloads/evaluate/metrics/cer/cer.py")

    # create a single speech processor
    if accelerator.is_local_main_process:
        # save feature extractor, tokenizer and config
        feature_extractor.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        model_config.save_pretrained(args.output_dir)

    # since tokenizer saved in args.output_dir
    processor = AutoProcessor.from_pretrained(args.output_dir)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task="transcribe")
    teacher.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task="transcribe")


    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    # data loaders
    train_dataloader = DataLoader(
        raw_datasets["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
    )
    eval_dataloader = DataLoader(
        raw_datasets["validation"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )
    test_dataloader = DataLoader(
        raw_datasets["test"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )

    # optimizer
    optimizer = AdamW(
        list(model.parameters()),
        lr=args.lr,
    )

    # calculate epochs
    #args.num_train_epochs = args.train_steps // len(train_dataloader) + 1

    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps
    )

    # prepare everything for accelerator
    # any instruction using your training dataloader length,
    # for instance if you need the number of total training steps
    # to create a learning rate scheduler) should go after the call to prepare()
    model, teacher, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, teacher, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval
    total_s_loss = 0  # total student loss before each eval
    total_d_loss = 0  # total distil loss before each eval

    # load from checkpoint
    ## loading checkpoint changing CER. val loss behaviour same. not sure why. ##
    # check if checkpoint directory passed in
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # if resumed from checkpoint
        # we need to skip steps until we reach the current step
        if args.skip_steps:
            # ../checkpoint-123 -> int(123)
            steps_completed = int(args.resume_from_checkpoint.split('/')[-1].split('-')[-1])
            train_dataloader = accelerator.skip_first_batches(train_dataloader, steps_completed) # consider dataset len
            global_step = steps_completed


    # Training

    # main progress bar
    progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process)

    while True:

        model.train()

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # student
                outputs = model(**batch)
                # stduent logits
                s_logits = outputs.logits
                # student loss
                s_loss = outputs.loss
                # teacher
                with torch.no_grad():
                    outputs = teacher(**batch)
                    # teacher logits
                    t_logits = outputs.logits
                # distillation loss
                # has to be outside no_grad()
                d_loss = nn.functional.kl_div(
                    input=nn.functional.log_softmax(s_logits / args.temperature, dim=-1),
                    target=nn.functional.softmax(t_logits / args.temperature, dim=-1),
                    reduction="batchmean",
                ) * (args.temperature**2)
                # net loss after weightage
                loss = args.alpha_distil * d_loss + args.alpha_ce * s_loss

                total_loss += loss.detach().item() # for tensorboard
                total_s_loss += s_loss.detach().item()
                total_d_loss += d_loss.detach().item()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)


            if (global_step + 1) % args.eval_steps == 0:
                model.eval()
                val_loss = 0
                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    # compute metric
                    ## check cer calculation ##
                    pred_logits = outputs.logits
                    pred_logits, references = accelerator.gather_for_metrics((pred_logits, batch["labels"]))
                    predictions = np.argmax(pred_logits.detach().cpu().clone().numpy(), axis=-1)
                    #predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                    references[batch['labels'] == -100] = processor.tokenizer.pad_token_id
                    predictions = processor.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    # we do not want to group tokens when computing the metrics
                    references = processor.batch_decode(references, group_tokens=False, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    metric.add_batch(predictions=predictions, references=references)

                cer_result = metric.compute()
                accelerator.print('step : {}, cer : {}'.format(global_step + 1, cer_result))
                accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                accelerator.log({
                    "cer": cer_result,
                    # might be incorrect
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    "train_s_loss": total_s_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    "train_d_loss": total_d_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    #"step": global_step,
                    "val_loss": val_loss / len(eval_dataloader)
                },
                step=global_step + 1,
                )

                # save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # saved to folders named `checkpoint-{global_step}`
                # will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # if mixed precision was used, will also save a "scalar.bin" file
                output_dir = f"checkpoint-{global_step + 1}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # save config
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    #model.config.save_pretrained(output_dir)
                    unwrapped_model.config.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

                model.train()
                total_loss = 0

            global_step += 1

            if global_step >= args.train_steps : return





def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,  # openai/whisper-small
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
        help="Path to trained teacher model",
    )
    parser.add_argument(
        "--alpha_ce",
        default=0.5,
        type=float,
        help="Cross entropy loss linear weight (student loss). Only for distillation."
    )
    parser.add_argument(
        "--alpha_distil",
        default=0.5,
        type=float,
        help="Distillation loss linear weight (distil loss). Only for distillation."
    )
    parser.add_argument(
        "--temperature",
        default=2.0,
        type=float,
        help="Distillation temperature. Only for distillation."
    )
    parser.add_argument(
        "--data_dir",
        default=None,  # mozilla-foundation/common_voice_11_0"
        type=str,
        help="Dataset",
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--max_eval_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--audio_column',
        type=str,
        default="audio",
        help="The name of the dataset column containing the audio data. Defaults to audio for cv."
    )

    parser.add_argument(
        '--text_column',
        type=str,
        default="transcript",
        help="The name of the dataset column containing the text data. Defaults to sentence for cv."
    )
    parser.add_argument(
        "--sampling_rate",
        default=16000,
        type=int,
        help="sampling rate",
    )
    parser.add_argument(
        "--output_dir",
        default=None,  # root+'/models/whisper/'+'whisper_small_cv11'
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="checkpoint directory to load model from",
    )
    parser.add_argument(
        "--skip_steps",
        action="store_true",
        help="whether to skip steps already ccompleted while loading from checkpoint"
    )
    parser.add_argument(
        "--lang",
        default='zh',
        type=str,
    )
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--mixed_precision",
        default='fp16',
        type=str,
    )



    # parse args
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # check if data path exists
    if args.data_dir is None:
        raise ValueError(
            f"pass in dataset directory"
        )
    if not os.path.isdir(args.data_dir):
        raise ValueError(
            f"data directory does not exist"
        )
    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )
    # check if model path is None
    if args.teacher_name_or_path is None:
        raise ValueError(
            f"pass in teacher_name_or_path"
        )
    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        teacher_str = args.teacher_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str+'_distil_from_'+teacher_str
    print('output directory set to : {}'.format(args.output_dir))
    

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=args.output_dir
    )
    # to have only one message per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.lr,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
    }
    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    main()

