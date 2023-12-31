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
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from tqdm.auto import tqdm
import argparse

import torch
from torch.utils.data.dataloader import DataLoader

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    GenerationConfig,
    get_linear_schedule_with_warmup, 
    set_seed
)
from modeling_whisper_moe import WhisperForConditionalGeneration

from datasets import (
    load_dataset, 
    load_from_disk, 
    DatasetDict, 
    Audio,
)

import evaluate

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


class MOETrainer:

    def __init__(
            self,
            args=None,
            model=None,
            tokenizer=None,
            processor=None,
            accelerator=None,
            data_collator=None,
            train_dataset=None,
            eval_dataset=None,
            metrics=None,
            **kwargs,
        ):
        
        self.args = args

        self.train_steps = args.train_steps
        self.global_step = 0  # tracks total steps

        self.model = model

        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.processor = processor

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        self.optimzer = None
        self.lr_scheduler = None

        self.metrics = metrics


    def create_optimizer_and_scheduler(self, args):
        # prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # optimzer
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.lr)

        # scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.train_steps
        )


    def train_step(self, inputs):
        
        with self.accelerator.accumulate(self.model):
            outputs = self.model(**inputs)
            loss = outputs.loss
            train_loss = loss.detach().item()  # for tensorboard
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {'train_loss': train_loss}
    

    def prediction_step(self, inputs, generation_config, gen_kwargs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            val_loss = outputs.loss.item()

        # compute metric
        # generate and calculate cer
        output_ids = self.accelerator.unwrap_model(self.model).generate(
            inputs["input_features"],
            generation_config=generation_config,
            task=self.args.task,
            language=self.args.model_lang,
            is_multilingual=True,
            **gen_kwargs
        )

        # pad_acrss_processes to get equal length for each processs
        output_ids = self.accelerator.pad_across_processes(
            output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
        label_ids = self.accelerator.pad_across_processes(
            inputs["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)

        # .cpu().numpy()  # gather_for_metrics
        output_ids = self.accelerator.gather(output_ids)
        # .cpu().numpy()  # gather_for_metrics
        label_ids = self.accelerator.gather(label_ids)

        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        predictions = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # we do not want to group tokens when computing the metrics
        references = self.processor.batch_decode(
            label_ids,
            group_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for key, val in self.metrics.items():
            val.add_batch(predictions=predictions, references=references)

        return val_loss


    def prediction_loop(self, dataloader, generation_config, gen_kwargs, description):

        # disable output hidden states and attention during evaluation
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False
        eval_loss = 0

        self.model.eval()

        # eval bar
        # eval_bar = tqdm(range(len(eval_dataloader)), position=1)
        for _, inputs in enumerate(tqdm(dataloader, desc=description)):
            # prediction step
            # metric added to self.metrics
            eval_loss += self.prediction_step(inputs, generation_config, gen_kwargs)

        results = {}

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        results['eval loss'] = eval_loss/len(dataloader)

        return results
    

    def evaluate(self, eval_dataloader, generation_config, gen_kwargs):

        results = self.prediction_loop(
            eval_dataloader, generation_config, gen_kwargs, description="Evaluation")

        self.accelerator.print('results : {}'.format(results))
        self.accelerator.log(results, step=self.global_step + 1)

        # save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
        # saved to folders named `checkpoint-{global_step}`
        # will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
        # if mixed precision was used, will also save a "scalar.bin" file
        output_dir = f"checkpoint-{self.global_step + 1}"
        if self.args.output_dir is not None:
            output_dir = os.path.join(self.args.output_dir, output_dir)
            # only saves weights, not model config
            self.accelerator.save_state(output_dir)
            # save config
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.config.save_pretrained(
                output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )


    def train(self, args, accelerator):

        # data loaders
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=args.train_batch_size,
        )
        eval_dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=self.data_collator,
            batch_size=args.eval_batch_size,
        )

        self.create_optimizer_and_scheduler(args)

        # prepare everything for accelerator
        # any instruction using your training dataloader length,
        # for instance if you need the number of total training steps
        # to create a learning rate scheduler) should go after the call to prepare()
        self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )

        accelerator.log({
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gpus": accelerator.state.num_processes
            },
            step=self.global_step + 1,
        )

        # load from checkpoint
        ## loading checkpoint changing CER. val loss behaviour same. not sure why. ##
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

        def make_generation_config():

            generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
            gen_dict = generation_config.to_dict()
            # add attributes to genration_config
            # generation_config does not have "langauge", but generate() tries to use it
            # can be empty dict here since being set in generate_step
            gen_dict["language"] = args.model_lang
            generation_config = GenerationConfig.from_dict(gen_dict)

            return generation_config


        max_length = (
            args.generation_max_length if args.generation_max_length is not None else self.model.config.max_length
        )
        num_beams = args.num_beams if args.num_beams is not None else self.model.config.num_beams
        gen_kwargs = {"max_new_tokens": max_length, "num_beams": num_beams}
        # generation config
        generation_config = make_generation_config()


        # Training #

        # main progress bar
        progress_bar = tqdm(range(self.global_step, args.train_steps), disable=not accelerator.is_main_process, position=0)
        # eval bar
        eval_bar = tqdm(range(len(eval_dataloader)), position=1)

        tr_loss = 0

        while True:

            self.model.train()

            for batch in train_dataloader:

                losses = self.train_step(batch)
                tr_loss += losses['train_loss']

                progress_bar.update(1)


                if (self.global_step + 1) % args.eval_steps == 0:

                    # log train losses
                    tr_loss = tr_loss / (args.eval_steps * self.accelerator.state.num_processes * args.train_batch_size)
                    self.accelerator.print('step : {}'.format(self.global_step + 1))
                    self.accelerator.print('train_loss : {}'.format(tr_loss))
                    self.accelerator.log({
                        "train_loss": tr_loss,
                    },
                    step=self.global_step + 1,
                    )
                    # evaluate model
                    self.evaluate(eval_dataloader, generation_config, gen_kwargs)

                    tr_loss = 0

                self.global_step += 1
                if self.global_step >= self.train_steps : return


def run():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--local",
        action="store_true",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="openai/whisper-small",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--activation",
        default=None,
        type=str,
        help="change model activation function",
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
        '--num_workers',
        type=int,
        default=os.cpu_count(), # 1, None, 32
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=6000,
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
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
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
   # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str+'_moe_train'
    print('output directory set to : {}'.format(args.output_dir))
    

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.lr,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
    }
    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # we only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    # model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    model.config.suppress_tokens = []

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    if args.activation is not None:
        model.config.activation_function = args.activation
        accelerator.print('activation changed to {}'.format(model.config.activation_function))

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # dataset
    if args.local:
        common_voice = load_from_disk(args.data_dir)
    else:
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

    # return attention_mask anyway
    forward_attention_mask = True

    # other hyperparameters
    max_input_length = args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = args.min_duration_in_seconds * feature_extractor.sampling_rate
    model_input_name = feature_extractor.model_input_names[0]

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

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )
    
    # cer, wer
    if args.local:
        # change path
        cer = evaluate.load("/home/ujan/Downloads/evaluate/metrics/cer/cer.py")
        wer = evaluate.load("/home/ujan/Downloads/evaluate/metrics/wer/wer.py")
    else:
        cer = evaluate.load("cer")
        wer = evaluate.load("wer")

    moe_trainer = MOETrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        accelerator=accelerator,
        data_collator=data_collator,
        train_dataset=common_voice['train'],
        eval_dataset=common_voice['test'],
        metrics={'cer': cer, 'wer': wer},
    )

    # train function
    moe_trainer.train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    run()


