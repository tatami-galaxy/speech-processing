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
import numpy as np
from tqdm.auto import tqdm, trange
from datasets import load_dataset, DatasetDict
import transformers, datasets
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import get_linear_schedule_with_warmup
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_scheduler, set_seed
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedType


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    


def train(args, accelerator):
    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task="transcribe")

    # model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)

    ## save config ##


    # dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test", use_auth_token=True)

    with accelerator.main_process_first():
        # remove unused columns
        common_voice = common_voice.remove_columns(
            [
                "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
            ]
        )

        # resample to 16kHz
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=args.sampling_rate))


    # function to vectorize dataset
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    
    with accelerator.main_process_first():
        # vectorize dataset
        common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"]) #, num_proc=2)



    # cer metric
    metric = evaluate.load("cer")

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

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
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    ##
    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval

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
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().item() # for tensorboard 
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
                    predictions = processor.batch_decode(predictions, skip_special_tokens=True)
                    # we do not want to group tokens when computing the metrics
                    references = processor.batch_decode(references, group_tokens=False, skip_special_tokens=True)  #,clean_up_tokenization_spaces=True
                    metric.add_batch(predictions=predictions, references=references)

                cer_result = metric.compute()
                accelerator.print('step : {}, cer : {}'.format(global_step + 1, cer_result))
                accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                accelerator.log({
                    "cer": cer_result,
                    # might be incorrect
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
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
        default="openai/whisper-tiny",
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
        default=root+'/models/whisper/'+'whisper_tiny_cv11',
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
        "--model_lang",
        default='Hindi',
        type=str,
    )
    parser.add_argument(
        "--data_lang",
        default='hi',
        type=str,
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
        default=2000,
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
        default=200,
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
    #if args.data_dir is None:
        #raise ValueError(
            #f"pass in dataset directory"
        #)
    ##args.processed_data_dir = root+'/data/processed/'+args.processed_data_dir+'/'
    #if not os.path.isdir(args.data_dir):
        #raise ValueError(
            #f"data directory does not exist"
        #)

    # check if output directory is passed in
    #if args.output_dir is None:
        #model_str = args.model_name_or_path.split('/')[-1]
        #data_str = 'cv11'
        #args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str
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

