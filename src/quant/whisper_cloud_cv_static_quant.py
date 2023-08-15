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
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset, DatasetDict
from datasets import Audio

from whisper_traceable import WhisperForConditionalGeneration
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput

from torch.ao.quantization import get_default_qconfig
from quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping

import evaluate



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
        #label_features = [{"input_ids": feature["labels"]} for feature in features]
        #decoder_input_ids = torch.LongTensor([feature["decoder_input_ids"] for feature in features])
        # needs to be LongTensor, cast before torch.histc
        decoder_input_ids = torch.LongTensor([feature["decoder_input_ids"] for feature in features])

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        #labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        #labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        #if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            #labels = labels[:, 1:]

        #batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch
    

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")
    


def load_model(args):
    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    # model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    #model.config.forced_decoder_ids = None
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    model.config.suppress_tokens = []

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False


    model.to("cpu")


    return model, feature_extractor, tokenizer, processor



def load_data(args, model, feature_extractor, tokenizer):


    # dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation")
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test")

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
    # get attention mask for observers
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
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        #input_str = batch["sentence"]  # do lower
        #batch["labels"] = tokenizer(input_str).input_ids
        batch["decoder_input_ids"] = torch.tensor([model.config.decoder_start_token_id]).reshape(1, -1)
        return batch
    
    
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

    return common_voice, forward_attention_mask


def quantize(model, example_inputs, qconfig_mapping, concrete_args, dataloader, is_encoder=False, encoder_outputs_list=[]):

    example_inputs.pop('decoder_input_ids')
    model = prepare_fx(model, qconfig_mapping, example_inputs, concrete_args=concrete_args)

    #if not is_encoder:
        #print(model.code)
        #quit()

    # get node names (concrete_arg_1)
    update_dict = {}
    for key, val in concrete_args.items():
        new_key = key+'_1'
        update_dict[new_key] = val
    
    # update nodes
    for node in model.graph.nodes:
        if node.name in update_dict:
            node.update_arg(0, update_dict[node.name])

    model.recompile()

    if is_encoder:
        encoder_outputs_list = calibrate(model, dataloader, is_encoder=is_encoder)
        quantized_model = convert_fx(model)
        return quantized_model, encoder_outputs_list
    
    else:
        calibrate(model, dataloader, is_encoder=is_encoder, encoder_outputs_list=encoder_outputs_list)
        quantized_model = convert_fx(model)
        return quantized_model


# calibrate with generate? or train data?
def calibrate(model, data_loader, is_encoder=False, encoder_outputs_list=[]):

    #past_key_values = [(None, None)]*12

    #if is_encoder:
        # store encoder_outputs for calibrating decoder 
        #encoder_outputs_list = []
    # calibration progress bar
    eval_bar = tqdm(range(len(data_loader)), position=0)
    model.eval()
    with torch.no_grad():
        for batch in data_loader:

            if is_encoder:
                batch.pop('decoder_input_ids')
                encoder_outputs = BaseModelOutput(model(**batch))
                #print(encoder_outputs)
                #print(type(encoder_outputs))
                #quit()
                encoder_outputs_list.append(encoder_outputs)

            else:  # single pass through decoder, no generation
                encoder_outputs = encoder_outputs_list.pop(0)
                model(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=batch['decoder_input_ids'],
                    #past_key_values=past_key_values,
                )

            eval_bar.update(1)

    if is_encoder: return encoder_outputs_list
    else: return

    



def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
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
        default="mozilla-foundation/common_voice_11_0",
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
        '--num_workers',
        type=int,
        default=None, # os.cpu_count()
        help="The number of processes to use for the preprocessing."
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
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str
    print('output directory set to : {}'.format(args.output_dir))
    ##if not os.path.isdir(args.output_dir):
        ##os.mkdir(args.output_dir)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )

    # model
    model, feature_extractor, tokenizer, processor = load_model(args)

    # data
    common_voice, forward_attention_mask = load_data(args, model, feature_extractor, tokenizer)

    # metrics
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
    test_dataloader = DataLoader(
        common_voice["test"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )


    # config for static quantization
    qconfig = get_default_qconfig("x86")
    #print(qconfig)
    #quit()
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # example inputs to set up observers
    example_inputs = (next(iter(train_dataloader)))
    #print(example_inputs)

    # concrete args for tracing control flow
    encoder_concrete_args={
        'output_hidden_states': False,
        'output_attentions': False,
        'head_mask': None,
        'return_dict': True,
    }
    decoder_concrete_args={
        'output_hidden_states': False,
        'output_attentions': False,
        'return_dict': True,
        'use_cache': True, 
        'decoder_inputs_embeds': None,
        'past_key_values': None, # cant be None for fast generation
    }

    # quantize encoder
    encoder = model.get_encoder()
    encoder_example_inputs = example_inputs.copy()
    quantized_encoder, encoder_outputs_list = quantize(
        encoder, encoder_example_inputs, qconfig_mapping, encoder_concrete_args, test_dataloader,
        is_encoder=True,
    )
    
    # quantize model
    decoder_example_inputs = example_inputs.copy()
    quantized_model = quantize(
        model, decoder_example_inputs, qconfig_mapping, decoder_concrete_args, test_dataloader,
        encoder_outputs_list=encoder_outputs_list,
    )
    

    print("Size of model before quantization")
    print_size_of_model(model)
    print("Size of model after quantization")
    print_size_of_model(quantized_model)





            


if __name__ == "__main__":

    main()

