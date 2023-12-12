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

from torch.utils.data.dataloader import DataLoader
import torch
from torch import nn
from torch.optim import AdamW

from accelerate import Accelerator

from transformers import(
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperConfig,
    GenerationConfig,
    get_linear_schedule_with_warmup, 
    set_seed,
)

from datasets import(
    load_dataset,
    DatasetDict,
    Audio,
)

import evaluate

from modeling_whisper import SparseWhisperForConditionalGeneration
from l0_module import L0Module

from transformers.debug_utils import DebugUnderflowOverflow

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


class CoFiTrainer:
    def __init__(
            self,
            args = None,
            model = None,
            tokenizer = None,
            processor = None,
            l0_module=None,
            teacher_model=None,
            accelerator=None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            metrics = None,
            **kwargs,
    ):
        
        self.args = args

        self.start_prune = False
        self.prepruning_finetune_steps = args.prepruning_finetune_steps
        self.train_steps = args.train_steps
        self.global_step = 0  # tracks total steps

        self.model = model
        self.teacher_model = teacher_model 
        self.distil_type = args.distil_type
        self.distil_temperature = args.distil_temperature
        self.alpha_ce = args.alpha_ce
        self.alpha_distil = args.alpha_distil
        self.l0_module = l0_module

        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.processor = processor

        self.optimizer = None
        self.l0_optimizer = None
        self.lagrangian_optimizer = None
        self.lr_scheduler = None   

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator

        self.metrics = metrics


    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True):

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            freeze_keywords = ["embeddings"]

            main_model_params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
            ]
            #log_params(main_model_params, "main params")
            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

            if build_l0_optimizer and self.l0_module is not None:
                l0_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": self.args.reg_learning_rate
                }]
                #log_params(l0_params, "l0 reg params")
                self.l0_optimizer = AdamW(l0_params,
                                          betas=(self.args.adam_beta1,
                                                 self.args.adam_beta2),
                                          eps=self.args.adam_epsilon, )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": -self.args.reg_learning_rate
                }]
                #log_params(lagrangian_params, "l0 reg lagrangian params")
                self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                    betas=(self.args.adam_beta1,
                                                            self.args.adam_beta2),
                                                    eps=self.args.adam_epsilon)

        if self.lr_scheduler is None:
            # scheduler_type == "linear"
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
            #else:
                #self.lr_scheduler = None


    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]


    def train_step(self, inputs):

        self.model.train()

        if self.l0_module is not None:
            self.l0_module.train()

        with self.accelerator.accumulate(self.model):
                
                d_loss = None
                lagrangian_loss = None

                if self.teacher_model is not None:
                    if self.distil_type == 'logit':           
                        outputs = self.model(**inputs)
                        # student logits
                        s_logits = outputs.logits
                        # student loss
                        s_loss = outputs.loss
                        # teacher
                        with torch.no_grad():
                            outputs = self.teacher_model(
                                input_features=inputs['input_features'],
                                attention_mask=inputs['attention_mask'],
                                labels=inputs['labels']
                            )
                        # teacher logits
                        t_logits = outputs.logits
                        # distillation loss
                        d_loss = nn.functional.kl_div(
                            input=nn.functional.log_softmax(s_logits / self.distil_temperature, dim=-1),
                            target=nn.functional.softmax(t_logits / self.distil_temperature, dim=-1),
                            reduction="batchmean",
                        ) * (self.distil_temperature**2)
                        # net loss after weightage
                        loss = self.alpha_distil * d_loss + self.alpha_ce * s_loss

                    elif self.distil_type == 'rail':
                        pass
                else:
                    outputs = self.model(**inputs)  # make sure model takes zs
                    loss = outputs.loss

                if self.start_prune:
                    # lagrangian_loss, expected_sparsity, target_sparsity
                    lagrangian_loss, _, _ = self.l0_module.lagrangian_regularization(self.global_step - self.prepruning_finetune_steps)
                    loss += lagrangian_loss

                # backward
                self.accelerator.backward(loss)

                # clip grad norm
                # error : clip_grad_norm_ returns inf
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                # step
                self.optimizer.step()
                if self.l0_module is not None and self.l0_optimizer is not None:
                    self.l0_optimizer.step()
                    self.lagrangian_optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # l0 constrain_parameters
                if self.l0_module is not None:
                    self.l0_module.constrain_parameters()

                # zero grad
                self.model.zero_grad()
                if self.l0_module is not None:
                    self.l0_module.zero_grad()
                self.optimizer.zero_grad()
                if self.l0_optimizer is not None:
                    self.l0_optimizer.zero_grad()
                if self.lagrangian_optimizer is not None:
                    self.lagrangian_optimizer.zero_grad()

                ret_dict = {}
                ret_dict['train_loss'] = loss.detach().item()
                if lagrangian_loss is not None:
                    ret_dict['lag_loss'] = lagrangian_loss.detach().item()
                if d_loss is not None:
                    ret_dict['distil_loss'] = d_loss.detach().item()
                # other losses (distill loss)

                return ret_dict



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
        output_ids = self.accelerator.pad_across_processes(output_ids, dim=1, pad_index=self.tokenizer.pad_token_id)
        label_ids = self.accelerator.pad_across_processes(inputs["labels"], dim=1, pad_index=self.tokenizer.pad_token_id)

        output_ids = self.accelerator.gather(output_ids)  #.cpu().numpy()  # gather_for_metrics
        label_ids = self.accelerator.gather(label_ids)  #.cpu().numpy()  # gather_for_metrics
                        
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        predictions = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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

        zs = None
        if self.start_prune:
            self.l0_module.eval()
            # real masks
            zs = self.l0_module.forward(training=False)  # contains zeros

        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs)

        # eval bar
        #eval_bar = tqdm(range(len(eval_dataloader)), position=1)
        for ii, inputs in enumerate(tqdm(dataloader, desc=description)):
            if zs is not None:
                #if ii == 0:
                    #logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) # use the zs
            # prediction step
            # metric added to self.metrics
            eval_loss += self.prediction_step(inputs, generation_config, gen_kwargs)

        results = {}

        if zs is not None: 
            lag_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)

            expected_sparsity = round(expected_sparsity.item(), 5)
            results.update(pruned_model_size_info)
            #results['lag_loss'] = lag_loss
            results["expected_sparsity"] = expected_sparsity
            results["target_sparsity"] = target_sparsity

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True    

        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        results['eval loss'] = eval_loss/len(dataloader)

        return results
    

    def evaluate(self, eval_dataloader, generation_config, gen_kwargs):

        results = self.prediction_loop(eval_dataloader, generation_config, gen_kwargs, description="Evaluation")

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
            # save zs and l0 module
            if self.l0_module is not None:
                zs = self.l0_module.forward(training=False)
                torch.save(zs, os.path.join(output_dir, 'zs_'+str(self.global_step+1)+'.pt'))
                torch.save(self.l0_module, os.path.join(output_dir, 'l0_module'+str(self.global_step+1)+'.pt'))



    def train(self, args):

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

        self.l0_module.set_lagrangian_warmup_steps(args.lagrangian_warmup_steps)

        self.create_optimizer_and_scheduler(num_training_steps=self.train_steps, build_l0_optimizer = self.start_prune)

        # model
        #model = self.model

        # prepare everything for accelerator
        # any instruction using your training dataloader length,
        # for instance if you need the number of total training steps
        # to create a learning rate scheduler) should go after the call to prepare()
        # works with none teacher
        self.model, self.teacher_model, self.l0_module, self.optimizer, self.l0_optimizer, self.lagrangian_optimizer, train_dataloader, eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.teacher_model, self.l0_module, self.optimizer, self.l0_optimizer, self.lagrangian_optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )

        self.accelerator.log({
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gpus": self.accelerator.state.num_processes
            },
            step=self.global_step + 1,
        )

        # load from checkpoint
        ## loading checkpoint changing CER. val loss behaviour same. not sure why. ##
        # check if checkpoint directory passed in
        if args.resume_from_checkpoint is not None:
            self.accelerator.print(f"resumed from checkpoint: {args.resume_from_checkpoint}")
            self.accelerator.load_state(args.resume_from_checkpoint)
            # if resumed from checkpoint
            # we need to skip steps until we reach the current step
            # ../checkpoint-123 -> int(123)
            steps_completed = int(args.resume_from_checkpoint.split('/')[-1].split('-')[-1])
            self.global_step = steps_completed
            if args.skip_steps:
                train_dataloader = self.accelerator.skip_first_batches(train_dataloader, steps_completed) # consider dataset len



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
            args.generation_max_length if args.generation_max_length is not None else self.model.config.max_length
        )
        num_beams = args.num_beams if args.num_beams is not None else self.model.config.num_beams
        gen_kwargs = {"max_new_tokens": max_length, "num_beams": num_beams}
        # generation config
        generation_config = make_generation_config()


        # Training

        #tr_loss = torch.tensor(0.0).to(self.args.device)
        #reg_loss = torch.tensor(0.0).to(self.args.device)
        #lag_loss = torch.tensor(0.0).to(self.args.device)

        self.model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        # main progress bar
        progress_bar = tqdm(range(self.global_step, self.train_steps), disable=not self.accelerator.is_main_process, position=0)

        tr_loss = 0  # train loss before each eval
        lag_loss = 0  # lagrangian loss before each eval

        debug_overflow = DebugUnderflowOverflow(self.model)

        while True:

            for batch in train_dataloader:
                #if self.prepruning_finetune_steps > 0 and self.global_step == self.prepruning_finetune_steps: #! before pruning, run 12272 steps
                if self.global_step == self.prepruning_finetune_steps: #! before pruning, run 12272 steps
                    self.start_prune = True

                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.train_steps - self.global_step

                    # reset the optimizer before strarting pruning
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    self.accelerator.print("starting l0 regularization")

                if self.start_prune:
                    # zs
                    # hidden, en_head, en_mha, en_ffn_dim, en_ffn, de_head, de_mha, de_ffn_dim, de_ffn
                    zs = self.l0_module.forward(training=True) # get the zs
                    # modifies batch in place
                    self.fill_inputs_with_zs(zs, batch) # use the zs

                # train step
                # recieve distill loss 
                losses = self.train_step(batch)
                tr_loss +=losses['train_loss']
                if 'lag_loss' in losses:
                    lag_loss += losses['lag_loss']
                # distil loss

                progress_bar.update(1)


                if (self.global_step + 1) % args.eval_steps == 0:

                    # log train losses
                    tr_loss = tr_loss / (args.eval_steps * self.accelerator.state.num_processes * args.train_batch_size)
                    lag_loss = lag_loss / (args.eval_steps * self.accelerator.state.num_processes * args.train_batch_size)
                    # distill loss #

                    self.accelerator.print('step : {}'.format(self.global_step + 1))
                    self.accelerator.print('train_loss : {}'.format(tr_loss))
                    self.accelerator.print('lag_loss : {}'.format(lag_loss))

                    self.accelerator.log({
                        "train_loss": tr_loss,
                        "lag_loss": lag_loss,
                        # distil loss #
                    },
                    step=self.global_step + 1,
                    )

                    self.evaluate(eval_dataloader, generation_config, gen_kwargs)


                self.global_step += 1

                if self.global_step >= self.train_steps : return





def run():


    parser = argparse.ArgumentParser()

    # seed
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )

    # whisper args
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
        "--learning_rate",
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
        "--adam_beta1",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--adam_beta2",
        default=0.999,
        type=float,
    )
    parser.add_argument(
        "--mixed_precision",
        default='no',
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

    # cofi args
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--distil_type",
        default=None,
        type=str,
    )
    parser.add_argument(
        '--droprate_init',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--l0_temperature',
        type=float,
        default=2./3.,
    )
    parser.add_argument(
        "--distil_temperature",
        default=2.0,
        type=float,
        help="distillation temperature"
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
        '--target_sparsity',
        type=float,
        default=0.95,
    )
    parser.add_argument(
        '--prepruning_finetune_steps',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--reg_learning_rate',
        type=float,
        default=0.1,
        help="learning rate for regularization."
    )
    parser.add_argument(
        '--lagrangian_warmup_steps',
        type=int,
        default=0,
    )
    parser.add_argument(
        "--start_prune",
        action="store_true",
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=1.0, # 1.0
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
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str+'_cofi'
    print('output directory set to : {}'.format(args.output_dir))
    # check distilation type
    if args.distil_type not in [None, 'logit', 'rail']:
        raise ValueError(
            f"distil_type must be either None or in [logit, rail]"
        )
    elif args.distil_type is not None:
        if args.teacher_name_or_path is None:
            raise ValueError(
                f"need to pass in teacher_name_or_path for distillation"
            )

    # accelerator mixed precision
    print('mixed precision set to : {}. fp16, fp8 may cause overflow/underflow'.format(args.mixed_precision))
    
    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.learning_rate,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
    }
    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # extractor, tokenizer, processor
    accelerator.print('loading tokenizer and processor')
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    if args.teacher_name_or_path is not None:
        accelerator.print('using same tokenizer and processor for teacher. change if needed')

    config = WhisperConfig.from_pretrained(
        args.model_name_or_path,
    )

    # working. detect sparse activation. change hardcoded gelus to relu
    if args.activation is not None:
        config.activation_function = args.activation
        accelerator.print('activation changed to {}'.format(config.activation_function))

    # model
    accelerator.print('loading model')
    model = SparseWhisperForConditionalGeneration.from_pretrained(
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

    # teacher #
    teacher_model = None
    if args.teacher_name_or_path is not None and args.distil_type is not None:
        accelerator.print('loading teacher')
        teacher_model = WhisperForConditionalGeneration.from_pretrained(args.teacher_name_or_path)
        teacher_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
        teacher_model.config.suppress_tokens = []
        if teacher_model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # l0 model
    l0_module = L0Module(
        config=config,
        droprate_init=args.droprate_init,
        temperature=args.l0_temperature,
        target_sparsity=args.target_sparsity,
    )
        

    # dataset

    if args.local:
        common_voice = load_dataset(args.data_dir, args.data_lang)
        # print(common_voice)
        # quit()
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

    # cer
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")


    cofi_trainer = CoFiTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        l0_module=l0_module,
        teacher_model=teacher_model,
        accelerator=accelerator,
        data_collator=data_collator,
        train_dataset=common_voice['train'],
        eval_dataset= common_voice['test'],
        metrics={'cer': cer, 'wer': wer},
    )

    accelerator.print('training')
    # train function
    cofi_trainer.train(args)

    # make sure model is saved

    # end logging
    accelerator.end_training()

    # run structural prune code (separate file)
            


if __name__ == "__main__":

    run()
