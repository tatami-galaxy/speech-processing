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
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import argparse

import transformers, datasets

from datasets import(
    load_dataset,
    DatasetDict,
    Audio,
)

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    GenerationConfig,
    get_linear_schedule_with_warmup,
    AdamW,
    set_seed
)
from whisper_traceable_masked import MaskedWhisperForConditionalGeneration
from configuration_whisper import MaskedWhisperConfig

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import evaluate

from accelerate import Accelerator


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    

def schedule_threshold(
    step: int,
    total_step: int,
    warmup_steps: int,
    initial_threshold: float,
    final_threshold: float,
    initial_warmup: int,
    final_warmup: int,
    final_lambda: float,
):
    if step <= initial_warmup * warmup_steps:
        threshold = initial_threshold
    elif step > (total_step - final_warmup * warmup_steps):
        threshold = final_threshold
    else:
        spars_warmup_steps = initial_warmup * warmup_steps
        spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff**3)
    regu_lambda = final_lambda * threshold / final_threshold
    return threshold, regu_lambda


def regularization(model: nn.Module, mode: str):
    regu, counter = 0, 0
    for name, param in model.named_parameters():
        if "mask_scores" in name:
            if mode == "l1":
                regu += torch.norm(torch.sigmoid(param), p=1) / param.numel()
            elif mode == "l0":
                regu += torch.sigmoid(param - 2 / 3 * np.log(0.1 / 1.1)).sum() / param.numel()
            else:
                ValueError("Don't know this mode.")
            counter += 1
    return regu / counter


## flatten mask after training
## store final threshold value?
## store final mask?
## check sparsity
## block sparsity
## how to remove matrices after block sparsity?
## relation to MOE?
## combine with static quantization
    


def train(args, accelerator):

    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    config = MaskedWhisperConfig.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir if args.cache_dir else None,
        pruning_method=args.pruning_method,
        mask_init=args.mask_init,
        mask_scale=args.mask_scale,
    )
    # model
    model = MaskedWhisperForConditionalGeneration.from_pretrained(
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



    # dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation")
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test")

    with accelerator.main_process_first():
        # remove unused columns
        common_voice = common_voice.remove_columns(
            [
                "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
            ]
        )

        # select small dataset for testing
        if args.max_train_samples is not None:
            common_voice["train"] = common_voice["train"].select(range(args.max_train_samples))

        if args.max_test_samples is not None:
            common_voice["test"] = common_voice["test"].select(range(args.max_test_samples))

        # resample to 16kHz
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=args.sampling_rate))


    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    #forward_attention_mask = (
        #getattr(model.config, "model_type", None) == "whisper"
        #and getattr(model.config, "apply_spec_augment", False)
        #and getattr(model.config, "mask_time_prob", 0) > 0
    #)
    # return attention_mask anyway
    forward_attention_mask = True

    # other hyperparameters
    max_input_length = args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = args.min_duration_in_seconds * feature_extractor.sampling_rate
    #audio_column_name = args.audio_column_name
    #num_workers = args.num_workers
    #text_column_name = args.text_column_name
    #do_lower_case = args.do_lower_case
    model_input_name = feature_extractor.model_input_names[0]

    # function to vectorize dataset
    #def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        #audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        #features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask=True)
        #batch["input_features"] = features.input_features[0]
        #batch["attention_mask"] = features.attention_mask[0]

        # encode target text to label ids 
        #batch["labels"] = tokenizer(batch["sentence"]).input_ids

        #return batch
    
    def prepare_dataset(batch):
        # process audio
        sample = batch["audio"]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask: # True, or check if needed above
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch["sentence"]  # do lower
        batch["labels"] = tokenizer(input_str).input_ids
        return batch
    
    
    with accelerator.main_process_first():
        # vectorize dataset
        common_voice = common_voice.map(
            prepare_dataset,
            remove_columns=common_voice.column_names["train"],
            num_proc=args.num_workers)


    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    common_voice = common_voice.filter(
        is_audio_in_length_range,
        num_proc=args.num_workers,
        input_columns=["input_length"],
    )



    # cer and wer
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    # data loaders
    train_dataloader = DataLoader(
        common_voice["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
    )
    eval_dataloader = DataLoader(
        common_voice["test"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "mask_score" in n and p.requires_grad],
            "lr": args.mask_scores_learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": args.lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

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
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    global_step = 0  # tracks total steps
    if args.global_topk:
        threshold_mem = None
    total_loss = 0  # total loss before each eval

    accelerator.log({
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gpus": accelerator.state.num_processes
        },
        step=global_step + 1,
    )

    # load from checkpoint
    # check if checkpoint directory passed in
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # if resumed from checkpoint
        # we need to skip steps until we reach the current step
            # ../checkpoint-123 -> int(123)
        steps_completed = int(args.resume_from_checkpoint.split('/')[-1].split('-')[-1])
        global_step = steps_completed
        if args.skip_steps:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, steps_completed) # consider dataset len
            

    
    def make_generation_config(supress_en=False):

        generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
        gen_dict = generation_config.to_dict()
        # add attributes to genration_config
        # generation_config does not have "langauge", but generate() tries to use it
        # can be empty dict here since being set in generate_step
        gen_dict["language"] = args.model_lang
        if supress_en:
            # en tokens to suppress from multilingual vocab
            en_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")  # change if loaded locally
            suppress_en_list = []
            for key in en_tokenizer.encoder.keys():
                if key in tokenizer.encoder.keys() and key.isalpha():
                    suppress_en_list.append(key)
            # supress english tokens
            gen_dict['suppress_tokens'].extend(tokenizer.encode(suppress_en_list, add_special_tokens=False))
        # add any other args here   
        # reload with new attributes
        generation_config = GenerationConfig.from_dict(gen_dict)

        return generation_config


    max_length = (
        args.generation_max_length if args.generation_max_length is not None else model.config.max_length
    )
    num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_new_tokens": max_length, "num_beams": num_beams}
    # generation config
    generation_config = make_generation_config()


    # Training

    # main progress bar
    progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process, position=0)
    eval_bar = tqdm(range(len(eval_dataloader)), position=1)

    model.zero_grad()

    while True:

        model.train()

        for batch in train_dataloader:

            threshold, regu_lambda = schedule_threshold(
                step=global_step,
                total_step=args.train_steps,
                warmup_steps=args.warmup_steps,
                final_threshold=args.final_threshold,
                initial_threshold=args.initial_threshold,
                final_warmup=args.final_warmup,
                initial_warmup=args.initial_warmup,
                final_lambda=args.final_lambda,
            )

            # Global TopK
            if args.global_topk:
                if threshold == 1.0:
                    threshold = -1e2  # or an indefinitely low quantity
                else:
                    if (threshold_mem is None) or (global_step % args.global_topk_frequency_compute == 0):
                        # Sort all the values to get the global topK
                        concat = torch.cat(
                            [param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name]
                        )
                        n = concat.numel()
                        kth = max(n - (int(n * threshold) + 1), 1)
                        threshold_mem = concat.kthvalue(kth).values.item()
                        threshold = threshold_mem
                    else:
                        threshold = threshold_mem

            with accelerator.accumulate(model):

                batch["threshold"] = threshold
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().item() # for tensorboard 

                # Regularization
                if args.regularization is not None:
                    regu_ = regularization(model=model, mode=args.regularization)
                    loss = loss + regu_lambda * regu_

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()

            progress_bar.update(1)

            if (global_step + 1) % args.eval_steps == 0:
                # eval progress bar
                #eval_bar = tqdm(range(len(eval_dataloader)), position=1)

                # Global TopK
                if args.global_topk:
                    threshold_mem = None

                model.eval()
                val_loss = 0

                for batch in eval_dataloader:
                    with torch.no_grad():

                        #batch["threshold"] = args.final_threshold
                        batch["threshold"] = threshold

                        if args.global_topk:
                            if threshold_mem is None:
                                concat = torch.cat(
                                    [param.view(-1) for name, param in model.named_parameters() if "mask_scores" in name]
                                )
                                n = concat.numel()
                                kth = max(n - (int(n * args.final_threshold) + 1), 1)
                                threshold_mem = concat.kthvalue(kth).values.item()
                            batch["threshold"] = threshold_mem

                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    # compute metric
                    # generate and calculate cer, wer
                    output_ids = accelerator.unwrap_model(model).generate(
                        batch["input_features"],
                        generation_config=generation_config,
                        task=args.task,
                        language=args.model_lang,
                        is_multilingual=True,
                        #threshold = args.final_threshold
                        threshold=threshold,
                        **gen_kwargs
                    )
                    # pad_acrss_processes recursively pads the tensors 
                    # in a nested list/tuple/dictionary of tensors from all devices 
                    # to the same size so they can safely be gathered
                    output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
                    label_ids = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                    # gather from all devices
                    output_ids = accelerator.gather(output_ids)  #.cpu().numpy()  # gather_for_metrics
                    label_ids = accelerator.gather(label_ids)  #.cpu().numpy()  # gather_for_metrics
                    # decode ids
                    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                    predictions = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    # we do not want to group tokens when computing the metrics
                    references = processor.batch_decode(
                        label_ids,
                        group_tokens=False,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    cer_metric.add_batch(predictions=predictions, references=references)
                    wer_metric.add_batch(predictions=predictions, references=references)

                    eval_bar.update(1)

                eval_bar.refresh()
                eval_bar.reset()

                cer_result = cer_metric.compute()
                wer_result = wer_metric.compute()

                accelerator.print('step : {}, cer : {}, wer: {}'.format(global_step + 1, cer_result, wer_result))
                accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                accelerator.log({
                    "cer": cer_result,
                    "wer": wer_result,
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    "val_loss": val_loss / len(eval_dataloader),
                    "threshold": threshold,
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


    # model and data
    parser.add_argument(
        "--model_name_or_path",
        default="openai/whisper-small",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--data_dir",
        default="mozilla-foundation/common_voice_13_0",
        type=str,
        help="Dataset",
    )
    parser.add_argument(
        "--sampling_rate",
        default=16000,
        type=int,
        help="sampling rate",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
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
        help="whether to skip steps in dataloader (checkpoint)"
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
        "--data_lang",
        default='hi',
        type=str,
    )
    parser.add_argument(
        '--max_duration_in_seconds',
        type=float,
        default=20.0
    )
    parser.add_argument(
        '--min_duration_in_seconds',
        type=float,
        default=0.0
    )


    # train and eval
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=5000,
        type=int,
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
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
        '--num_workers',
        type=int,
        default=os.cpu_count(), # os.cpu_count()
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--mixed_precision",
        default='fp16',
        type=str,
    )
    parser.add_argument(
        '--generation_max_length',
        type=int,
        default=225
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=1
    )


    # prune
    parser.add_argument(
        "--mask_scores_learning_rate",
        default=1e-2,
        type=float,
        help="The Adam initial learning rate of the mask scores.",
    )
    parser.add_argument(
        "--initial_threshold", default=1.0, type=float, help="Initial value of the threshold (for scheduling)."
    )
    parser.add_argument(
        "--final_threshold", default=0.7, type=float, help="Final value of the threshold (for scheduling)."
    )
    parser.add_argument(
        "--initial_warmup",
        default=1,
        type=int,
        help=(
            "Run `initial_warmup` * `warmup_steps` steps of threshold warmup during which threshold stays"
            "at its `initial_threshold` value (sparsity schedule)."
        ),
    )
    parser.add_argument(
        "--final_warmup",
        default=2,
        type=int,
        help=(
            "Run `final_warmup` * `warmup_steps` steps of threshold cool-down during which threshold stays"
            "at its final_threshold value (sparsity schedule)."
        ),
    )

    parser.add_argument(
        "--pruning_method",
        default="topK",
        type=str,
        help=(
            "Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning,"
            " sigmoied_threshold = Soft movement pruning)."
        ),
    )
    parser.add_argument(
        "--mask_init",
        default="constant",
        type=str,
        help="Initialization method for the mask scores. Choices: constant, uniform, kaiming.",
    )
    parser.add_argument(
        "--mask_scale", default=0.0, type=float, help="Initialization parameter for the chosen initialization method."
    )

    parser.add_argument("--regularization", default=None, help="Add L0 or L1 regularization to the mask scores.")
    parser.add_argument(
        "--final_lambda",
        default=0.0,
        type=float,
        help="Regularization intensity (used in conjunction with `regularization`.",
    )

    parser.add_argument("--global_topk", action="store_true", help="Global TopK on the Scores.")
    parser.add_argument(
        "--global_topk_frequency_compute",
        default=25,
        type=int,
        help="Frequency at which we compute the TopK global threshold.",
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
    ##args.processed_data_dir = root+'/data/processed/'+args.processed_data_dir+'/'
    #if not os.path.isdir(args.data_dir):
        #raise ValueError(
            #f"data directory does not exist"
        #)

    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'-'+data_str+'-pruned'
    #print('output directory set to : {}'.format(args.output_dir))
    ##if not os.path.isdir(args.output_dir):
        ##os.mkdir(args.output_dir)

    # check if model path is None
    #if args.model_name_or_path is None:
        #raise ValueError(
            #f"pass in model_name_or_path"
        #)
    

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir  # project_dir 
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "seed": args.seed,
        "gpus": accelerator.state.num_processes,
        "lr": args.lr,
        "train_steps": args.train_steps,
        "eval_steps": args.eval_steps,
        "train_batch_size": args.train_batch_size,
        "initial_threshold": args.initial_threshold,
        "final_threshold": args.final_threshold,
        "initial_warmup": args.initial_warmup,
        "final_warmup": args.final_warmup,
        "pruning_method": args.pruning_method,
        "mask_init": args.mask_init,
        "mask_scale": args.mask_scale,
        "final_lambda": args.final_lambda,
    }

    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', config=track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    main()

