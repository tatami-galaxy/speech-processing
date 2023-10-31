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
from torch.optim import AdamW

from accelerate import Accelerator

from transformers import(
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperConfig,
    GenerationConfig,
    get_linear_schedule_with_warmup, 
    set_seed,
)
from transformers.modeling_utils import PreTrainedModel

from datasets import(
    load_dataset,
    DatasetDict,
    Audio,
)

import evaluate

from modeling_whisper import SparseWhisperForConditionalGeneration
from l0_module import L0Module

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
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            metrics = None,
            **kwargs,
    ):
        
        self.args = args

        self.start_prune = False
        self.prepruning_finetune_steps = args.prepruning_finetune_steps

        self.l0_module = l0_module
        self.l0_optimizer = None
        self.lagrangian_optimizer = None
        self.lr_scheduler = None
        self.model = model
        self.teacher_model = teacher_model    

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
                    "lr": self.additional_args.reg_learning_rate
                }]
                #log_params(l0_params, "l0 reg params")
                self.l0_optimizer = AdamW(l0_params,
                                          betas=(self.args.adam_beta1,
                                                 self.args.adam_beta2),
                                          eps=self.args.adam_epsilon, )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": -self.additional_args.reg_learning_rate
                }]
                #log_params(lagrangian_params, "l0 reg lagrangian params")
                self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                    betas=(self.args.adam_beta1,
                                                            self.args.adam_beta2),
                                                    eps=self.args.adam_epsilon)

        if self.lr_scheduler is None:
            if self.additional_args.scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None


    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]


    def train_step(self, model, inputs, accelerator):

        with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().item() # for tensorboard 
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                tr_loss_step = loss_terms["loss"]
                lag_loss_step = loss_terms["lagrangian_loss"]

                tr_loss += tr_loss_step
                lag_loss += lag_loss_step if lag_loss_step is not None else 0.0

                torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)

                self.optimizer.step()

                if self.l0_module is not None and self.l0_optimizer is not None:
                    self.l0_optimizer.step()
                    self.lagrangian_optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                if self.l0_module is not None:
                    self.l0_module.constrain_parameters()

                model.zero_grad()
                if self.l0_module is not None:
                    self.l0_module.zero_grad()
                self.optimizer.zero_grad()
                if self.l0_optimizer is not None:
                    self.l0_optimizer.zero_grad()
                if self.lagrangian_optimizer is not None:
                    self.lagrangian_optimizer.zero_grad()

                self.global_step += 1
                self.epoch = epoch + (step + 1) / len(epoch_iterator)

                if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                ):
                    logs: Dict[str, float] = {}
                    tr_loss_scalar = tr_loss.item()
                    reg_loss_scalar = reg_loss.item()
                    lag_loss_scalar = lag_loss.item()

                    logs["loss"] = (
                        tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                    logs["reg_loss"] = (
                        reg_loss_scalar - logging_reg_loss_scalar) / self.args.logging_steps
                    logs["lag_loss"] = (
                        lag_loss_scalar - logging_lag_loss_scalar) / self.args.logging_steps

                    # backward compatibility for pytorch schedulers
                    if self.lr_scheduler is not None:
                        lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                            torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                    else:
                        lr = self.args.learning_rate

                    logs["learning_rate"] = lr
                    logging_loss_scalar = tr_loss_scalar
                    logging_reg_loss_scalar = reg_loss_scalar
                    logging_lag_loss_scalar = lag_loss_scalar

                    self.log(logs)


    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[Dict[str, float], List]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        # wandb.log(output.metrics)
        output.metrics["step"] = self.global_step

        logger.info(f"Evaluating: {output.metrics}")

        eval_score = 0

        name = glue_tasks[self.model.config.finetuning_task]
        if isinstance(name, str):
            if name in output.metrics:
                eval_score = output.metrics[name]
        else:
            for na in name:
                if na in output.metrics:
                    eval_score = output.metrics[na]
                    break

        # logger.info(f"starting saving best: {self.global_step} {self.start_saving_best}")

        if self.start_saving_best:
            best_so_far = self.eval_counter.update(
                self.epoch, self.global_step, eval_score)
            if best_so_far:
                best_dir = os.path.join(self.args.output_dir, "best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)

                if self.l0_module is not None:
                    zs = self.l0_module.forward(training=False)
                    torch.save(zs, os.path.join(best_dir, "zs.pt"))
                    torch.save(self.l0_module, os.path.join(
                        best_dir, "l0_module.pt"))
                logger.info(f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.global_step} | Model size: {output.metrics['remaining_params'] if 'remaining_params' in output.metrics else 'Full' } | Score: {round(eval_score, 5)}]")
                self.model.save_pretrained(best_dir)

        return output.metrics
    

    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # disable output hidden states and attention during evaluation
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

        model = self.model

        # multi-gpu eval
        model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm

        zs = None
        if self.start_prune:
            self.l0_module.eval()
            zs = self.l0_module.forward(training=False)

        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs)

        for ii, inputs in enumerate(tqdm(dataloader, desc=description, disable=disable_tqdm)):
            if zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) #! use the zs
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only)

            batch_size = inputs[list(inputs.keys())[0]].shape[0]

            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels)
            if loss is not None:
                if type(loss) == float:
                    losses = [loss] * batch_size
                    if losses_host is None:
                        losses_host = losses
                    else:
                        losses_host.extend(losses)
                else:
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat(
                        (losses_host, losses), dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation.py loop
            delattr(self, "_past")

        if losses_host is not None:
            if not torch.is_tensor(losses_host):
                losses_host = torch.tensor(losses_host)
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(
                predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None and len(all_losses) > 0:
            metrics["eval_loss"] = np.mean(all_losses)

        if zs is not None:
            lag_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)

            expected_sparsity = round(expected_sparsity.item(), 5)
            metrics.update(pruned_model_size_info)
            metrics["expected_sparsity"] = expected_sparsity
            metrics["target_sparsity"] = target_sparsity

            if (not self.start_saving_best) and (expected_sparsity - self.additional_args.target_sparsity >= -self.additional_args.sparsity_epsilon):
                self.start_saving_best = True
                logger.info(f"Starting saving the best from epoch {int(self.epoch)} and step {self.global_step}")

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)


    def train(self, args, accelerator):

        # data loaders
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=args.train_batch_size,
        )
        eval_dataloader = DataLoader(
            self.test_dataset,
            collate_fn=self.data_collator,
            batch_size=args.eval_batch_size,
        )

        num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        lagrangian_warmup_steps = args.lagrangian_warmup_epochs * num_update_steps_per_epoch #! 24544
        # self.prepruning_finetune_steps = self.additional_args.prepruning_finetune_epochs * num_update_steps_per_epoch
        self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)

        self.create_optimizer_and_scheduler(num_training_steps=args.train_steps, build_l0_optimizer = self.start_prune)

        # model
        model = self.model

        # prepare everything for accelerator
        # any instruction using your training dataloader length,
        # for instance if you need the number of total training steps
        # to create a learning rate scheduler) should go after the call to prepare()
        model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler = accelerator.prepare(
            model, self.optimizer, train_dataloader, eval_dataloader, self.lr_scheduler
        )


        global_step = 0  # tracks total steps
        total_loss = 0  # total loss before each eval


        accelerator.log({
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gpus": accelerator.state.num_processes
            },
            step=global_step + 1,
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


        # Training

        tr_loss = torch.tensor(0.0).to(self.args.device)
        reg_loss = torch.tensor(0.0).to(self.args.device)
        lag_loss = torch.tensor(0.0).to(self.args.device)

        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        # main progress bar
        progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process, position=0)
        # eval bar
        eval_bar = tqdm(range(len(eval_dataloader)), position=1)

        while True:

            model.train()

            for batch in train_dataloader:

                if self.prepruning_finetune_steps > 0 and self.global_step == self.prepruning_finetune_steps: #! before pruning, run 12272 steps
                    self.start_prune = True

                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.global_step

                    # reset the optimizer
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    accelerator.print("starting l0 regularization")

                if self.start_prune:
                    zs = self.l0_module.forward(training=True) #! get the zs
                    self.fill_inputs_with_zs(zs, batch) #! use the zs

                loss_terms = self.train_step(model, batch, accelerator)


                progress_bar.update(1)


                if (global_step + 1) % args.eval_steps == 0:
                    model.eval()
                    val_loss = 0
                    for batch in eval_dataloader:
                        with torch.no_grad():
                            outputs = model(**batch)
                            val_loss += outputs.loss.item()

                        # compute metric
                        # generate and calculate cer 
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
                    accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                    accelerator.log({
                        "cer": cer_result,
                        "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
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





def run():


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
        '--target_sparsity',
        type=float,
        default=0,
    )
    parser.add_argument(
        '--prepruning_finetune_steps',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--lagrangian_warmup_epochs',
        type=int,
        default=2,
    )
    parser.add_argument(
        "--start_prune",
        action="store_true",
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
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str
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
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    config = WhisperConfig.from_pretrained(
        args.model_name_or_path,
    )

    # working. detect sparse activation. change hardcoded gelus to relu
    if args.activation is not None:
        config.activation_function = args.activation
        print('activation changed to {}'.format(config.activation_function))

    # model
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

    # distil setup #

    l0_module = L0Module(
        config=config,
        droprate_init=args.droprate_init,
        temperature=args.l0_temperature,
        target_sparsity=args.target_sparsity,
    )
        

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
        #teacher_model=teacher_model
        data_collator=data_collator,
        train_dataset=common_voice['train'],
        eval_dataset= common_voice['test'],
        metrics=[cer, wer],
    )

    # train function
    cofi_trainer.train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    run()

