"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

import re
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoConfig
from transformers import AutoFeatureExtractor
from transformers import AutoTokenizer
from transformers import AutoProcessor
from transformers import GenerationConfig
from typing import List
from transformers import AutoModelForSpeechSeq2Seq
from optimum.bettertransformer import BetterTransformer
from transformers import set_seed
import argparse
import timeit
# Intel Extension for PyTorch
import intel_extension_for_pytorch as ipex
# neural comressor
#from neural_compressor.compression.pruner import model_slim

#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=50000))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'



def train(args):

    # load dataset
    print('loading dataset from {}'.format(args.data_dir))

    raw_datasets = load_dataset(args.data_dir)

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

    if args.max_test_samples is not None:
        raw_datasets["test"] = raw_datasets["test"].select(range(args.max_test_samples))



     # remove punctuations
    def remove_special_characters(batch):
        batch[args.text_column] = re.sub(chars_to_ignore_regex, "", batch[args.text_column]).lower() + " "
        return batch
    
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


    model = AutoModelForSpeechSeq2Seq.from_pretrained(
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



    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(duration):
        return duration > args.min_duration and duration < args.max_duration

    raw_datasets = raw_datasets.filter(
        is_audio_in_length_range,
        num_proc=args.preprocessing_num_workers,
        input_columns=["duration"],
        #keep_in_memory=True
    )

    dataset = raw_datasets['test']


    # better transformer 
    model = BetterTransformer.transform(model, keep_original_model=False)


    # Intel Extention
    # Apply some fusions at the front end
    #model = ipex.optimize(model, dtype=torch.float32)
    #model = ipex.optimize(model)


    ## dynamic quant ##

    # The easiest method of quantization PyTorch supports is called dynamic quantization.
    # This involves not just converting the weights to int8 - as happens in all quantization variants -
    # but also converting the activations to int8 on the fly, just before doing the computation (hence “dynamic”).
    # The computations will thus be performed using efficient int8 matrix multiplication
    # and convolution implementations, resulting in faster compute.
    # However, the activations are read and written to memory in floating point format
    
    
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )



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


    #model = model_slim(model)

    # compile
    model = torch.compile(model)


    model.eval()

    # warm up for compile
    #warmup_samples = 10
    #warmup_dataset = dataset.select(range(warmup_samples))

    #print('warmup')
    #for sample in warmup_dataset:
        #inputs = processor(sample["audio"]["array"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        #input_features = inputs.input_features
        #output_ids = model.generate(
            #input_features,
            #generation_config=generation_config,
            #task=args.task,
            #language=args.model_lang,
            #is_multilingual=True,
            #**gen_kwargs
        #)
    #print('warmup done')

    # eval bar
    eval_bar = tqdm(range(len(dataset)), position=0)

    per_sec_inf_times = []

    for sample in dataset:
        inputs = processor(sample["audio"]["array"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        input_features = inputs.input_features
        # start timer
        start_time = timeit.default_timer()
        output_ids = model.generate(
            input_features,
            generation_config=generation_config,
            task=args.task,
            language=args.model_lang,
            is_multilingual=True,
            **gen_kwargs
        )
        predictions = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        # end timer
        elapsed = timeit.default_timer() - start_time
        per_sec_inf_times.append(elapsed / sample["duration"])
        eval_bar.update(1)


    print("average per sec inference time : {}".format(sum(per_sec_inf_times)/len(dataset)))




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
        "--data_dir",
        default=None,  # mozilla-foundation/common_voice_11_0"
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
        default=None,  # os.cpu_count(), # None, 32
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
        default='chinese',
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

    # train function
    train(args)



            


if __name__ == "__main__":

    run()

