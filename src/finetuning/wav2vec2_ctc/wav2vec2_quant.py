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
from transformers import Wav2Vec2ForCTC
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
#import intel_extension_for_pytorch as ipex
# neural comressor
#from neural_compressor.compression.pruner import model_slim

#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=50000))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'



def train(args):

    # load dataset
    print('loading dataset from {}'.format(args.data_dir))

    dataset = load_dataset(args.data_dir, "clean", split="validation")

    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate


    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name_or_path)



    # better transformer 
    #model = BetterTransformer.transform(model, keep_original_model=False)


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
    
    
    #model = torch.quantization.quantize_dynamic(
        #model, {torch.nn.Linear}, dtype=torch.qint8
    #)


    # compile
    #model = torch.compile(model)


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
        inputs = processor(sample["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
        duration = len(sample["audio"]["array"])/sampling_rate
        # start timer
        start_time = timeit.default_timer()
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        # end timer
        elapsed = timeit.default_timer() - start_time
        per_sec_inf_times.append(elapsed / duration)
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

