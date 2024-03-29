"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

import os, re
import datetime
import timeit
from os.path import dirname, abspath
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
import transformers, datasets
from transformers import AutoConfig
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import AutoProcessor
from transformers import GenerationConfig
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from whisper_pt_prune import WhisperForConditionalGeneration
from torch.utils.data.dataloader import DataLoader
from transformers import set_seed
import argparse
from accelerate import Accelerator

#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=50000))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


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
    accelerator.print('loading dataset from {}'.format(args.data_dir))

    # dataset
    raw_datasets = DatasetDict()
    raw_datasets["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation")
    raw_datasets["test"] = load_dataset(args.data_dir, args.data_lang, split="test")

    with accelerator.main_process_first():
        # remove unused columns
        raw_datasets = raw_datasets.remove_columns(
            [
                "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
            ]
        )

    # check audio column, text column names
    if args.audio_column not in raw_datasets["test"].column_names:
        raise ValueError(
            f"--audio_column '{args.audio_column}' not found in dataset '{args.data_dir}'."
            " Make sure to set `--audio_column` to the correct audio column - one of"
            f" {', '.join(raw_datasets['test'].column_names)}."
        )

    if args.text_column not in raw_datasets["test"].column_names:
        raise ValueError(
            f"--text_column {args.text_column} not found in dataset '{args.data_dir}'. "
            "Make sure to set `--text_column` to the correct text column - one of "
            f"{', '.join(raw_datasets['test'].column_names)}."
        )

    with accelerator.main_process_first():
        if args.max_test_samples is not None:
            raw_datasets["test"] = raw_datasets["test"].select(range(args.max_test_samples))



     # remove punctuations
    def remove_special_characters(batch):
        batch[args.text_column] = re.sub(chars_to_ignore_regex, "", batch[args.text_column]).lower() + " "
        return batch
    
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            remove_special_characters,
            desc="remove special characters from datasets",
        )


    # model, tokenizer, feature extractor, processor

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )

    model_config.update({"forced_decoder_ids": args.forced_decoder_ids, "suppress_tokens": args.suppress_tokens})


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
        tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)


    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        #cache_dir=args.cache_dir,
        #revision=args.model_revision,
        #use_auth_token=True if args.use_auth_token else None,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)


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
            num_proc=num_workers,
            #keep_in_memory=True,  # no cache
            desc="preprocess test dataset",
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
            #keep_in_memory=True
        )



    # cer metric
    metric = evaluate.load("cer")


    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )


    # data loader
    test_dataloader = DataLoader(
        vectorized_datasets["test"],
        collate_fn=data_collator,
        batch_size=args.test_batch_size,
    )

    # prepare everything for accelerator
    # any instruction using your training dataloader length,
    # for instance if you need the number of total training steps
    # to create a learning rate scheduler) should go after the call to prepare()
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    global_step = 0  # tracks total steps


    # load from checkpoint
    ## loading checkpoint changing CER. val loss behaviour same. not sure why. ##
    # check if checkpoint directory passed in
    if args.checkpoint is not None:
        accelerator.print(f"loaded from checkpoint: {args.checkpoint}")
        accelerator.load_state(args.checkpoint)



    def make_generation_config():

        generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
        gen_dict = generation_config.to_dict()
        # add attributes to genration_config
        # generation_config does not have "langauge", but generate() tries to use it
        # can be empty dict here since being set in generate_step
        gen_dict["language"] = args.model_lang
        #if supress_en:
            # en tokens to suppress from multilingual vocab
            #en_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")  # change if loaded locally
            #suppress_en_list = []
            #for key in en_tokenizer.encoder.keys():
                #if key in tokenizer.encoder.keys() and key.isalpha():
                    #suppress_en_list.append(key)
            # supress english tokens
            #gen_dict['suppress_tokens'].extend(tokenizer.encode(suppress_en_list, add_special_tokens=False))
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

    # prune zero heads
    if args.prune:
        model.prune_heads()
        accelerator.print('heads pruned')


    # eval bar
    eval_bar = tqdm(range(len(test_dataloader)), position=0)

    model.eval()
    val_loss = 0

    for batch in test_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            val_loss += outputs.loss.item()

        output_ids = accelerator.unwrap_model(model).generate(
            batch["input_features"],
            generation_config=generation_config,
            task=args.task,
            language=args.model_lang,
            is_multilingual=True,
            **gen_kwargs
        )

        # pad_acrss_processes to get equal length for each processs
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        label_ids = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

        output_ids = accelerator.gather(output_ids)  #.cpu().numpy()  # gather_for_metrics
        label_ids = accelerator.gather(label_ids)  #.cpu().numpy()  # gather_for_metrics
                    
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predictions = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # we do not want to group tokens when computing the metrics
        references = processor.batch_decode(
            label_ids,
            group_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        metric.add_batch(predictions=predictions, references=references)

        eval_bar.update(1)


    eval_bar.refresh()
    eval_bar.reset()

    cer_result = metric.compute()
    # add wer for hindi
    accelerator.print('step : {}, cer : {}'.format(global_step + 1, cer_result))
    accelerator.print('val loss : {}'.format(val_loss/len(test_dataloader)))
    accelerator.log({
        "cer": cer_result,
        "val_loss": val_loss / len(test_dataloader)
    },
    step=global_step + 1,
    )



def run():


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
        '--freeze_encoder',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to freeze the transformer encoder of the model."
    )
    parser.add_argument(
        '--prune',
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to prune heads of the model."
    )
    parser.add_argument(
        "--data_dir",
        default="mozilla-foundation/common_voice_13_0",  # mozilla-foundation/common_voice_11_0"
        type=str,
        help="Dataset",
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
        default="sentence",
        help="The name of the dataset column containing the text data. Defaults to sentence for cv."
    )
    parser.add_argument(
        "--sampling_rate",
        default=16000,
        type=int,
        help="sampling rate",
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=20.0,
        help="Filter audio files that are longer than max_duration."
    )

    parser.add_argument(
        '--min_duration',
        type=float,
        default=1.0, # 0.0
        help="Filter audio files that are shorter than min_duration."
    )
    parser.add_argument(
        '--preprocessing_num_workers',
        type=int,
        default=os.cpu_count(), # None, 32
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="checkpoint directory to load model from",
    )
    parser.add_argument(
        "--model_lang",
        default='hindi',
        type=str,
    )
    parser.add_argument(
        "--data_lang",
        default='hi',
        type=str,
    )
    parser.add_argument(
        "--task",
        default='transcribe',
        type=str,
    )
    parser.add_argument(
        "--test_batch_size",
        default=8,
        type=int,
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



    # parse args
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # check if data path exists
    if args.data_dir is None:
        raise ValueError(
            f"pass in dataset directory"
        )
    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir='./outputs'
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
        "seed": args.seed,
        "test_batch_size": args.test_batch_size,
    }
    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    run()

