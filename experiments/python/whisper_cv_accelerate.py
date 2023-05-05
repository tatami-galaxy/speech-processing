"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

import math
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
from transformers import AdamW, get_scheduler
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedType


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)


# set seed
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



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





def main():


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_name_or_path",
        default="openai/whisper-small",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--data_dir",
        default="mozilla-foundation/common_voice_11_0",
        type=str,
        required=True,
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
        default=root+'/models/whisper/'+'whisper_small_cv11',
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--model_lang",
        default='Hindi',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_lang",
        default='hi',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        required=True,
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
        default=None,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--mixed_precision",
        default=1e-5,
        type=float,
    )



    # parse args
    args = parser.parse_args()

    # set seed
    set_seed(args)

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


    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task="transcribe")

    # model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    # dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test", use_auth_token=True)

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
    # any instruction using your training dataloader length
    # (for instance if you need the number of total training steps
    # to create a learning rate scheduler) should go after the call to prepare()
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    ##
    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval

    # checkpointing -> load from checkpoint #


    # Training
    progress_bar = tqdm(args.train_steps, disable=not accelerator.is_main_process)
    while True:
        model.train()

        # if resumed from checkpoint
        # we need to skip steps until we reach the resumed step

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().item() # for tensorboard 
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            # checkpoint #


            if (global_step + 1) % args.eval_steps == 0:
                model.eval()
                val_loss = 0
                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    # compute metric
                    pred_logits = outputs.logits
                    predictions = np.argmax(pred_logits.detach().cpu().clone().numpy(), axis=-1)
                    predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                    references[batch['labels'] == -100] = processor.tokenizer.pad_token_id
                    predictions = processor.batch_decode(predictions)
                    # we do not want to group tokens when computing the metrics
                    references = processor.batch_decode(references, group_tokens=False)
                    metric.add_batch(predictions=predictions, references=references)

                cer_result = metric.compute()
                accelerator.print('step : {}, cer : {}'.format(step, cer_result))
                accelerator.print(val_loss/len(eval_dataloader))
                accelerator.log({
                    "cer": cer_result,
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    #"step": global_step,
                    "val_loss": val_loss / len(eval_dataloader)
                },
                step=global_step,
                )

               # accelerator.print('saving')
                #model.save_pretrained(args.output_dir+'/'+'checkpoint-'+str(step+1))

                model.train()
                total_loss = 0

            if step >= num_training_steps : break
            else: continue

        break

            


if __name__ == "__main__":

    main()

