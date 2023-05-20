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
from pathlib import Path
from functools import partial
import logging
import argparse
from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass
from tqdm.auto import tqdm
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import (
    onehot,
    shard,
    shard_prng_key
)

import transformers
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    FlaxWhisperForConditionalGeneration
)
from transformers import (
    AdamW,
    set_seed,
    get_linear_schedule_with_warmup,
    is_tensorboard_available,
)
from transformers.utils import send_example_telemetry

import datasets
from datasets import load_dataset, DatasetDict
from datasets import Audio
import torch
import evaluate
from torch.utils.data.dataloader import DataLoader


logger = logging.getLogger(__name__)

# sending telemetry
# tracking the example usage helps us better allocate resources to maintain them
# the information sent is the one passed as arguments along with your Python/PyTorch versions
send_example_telemetry("run_summarization", framework="flax")


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)



@dataclass
class FlaxDataCollatorForWhisperFinetuning:
    processor: Any
    padding: Union[bool, str] = 'longest'
    pad_to_multiple_of: Optional[int] = None
    max_length: Optional[int] = None

    # features -> input_features, decoder_input_ids, labels
    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, np.ndarray]:

        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(  # feature extractor pad
            input_features,
            padding=self.padding,
            return_tensors="np")

        # get the tokenized decoder_input_ids
        decoder_features = [{"input_ids": feature["decoder_input_ids"][0]} for feature in features]
        # pad the labels to max length
        decoder_input_batch = self.processor.tokenizer.pad(  # tokenizer pad since text
            decoder_features,
            padding=self.padding,
            return_tensors="np")    
        # replace padding with -100 to ignore loss correctly
        #decoder_inputs = decoder_input_batch["input_ids"].masked_fill(decoder_input_batch.attention_mask.ne(1), -100)
        decoder_inputs = decoder_input_batch["input_ids"]
        decoder_attention = decoder_input_batch["attention_mask"]
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (decoder_inputs[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            decoder_inputs = decoder_inputs[:, 1:]
        batch["decoder_input_ids"] = decoder_inputs
        batch["decoder_attention_mask"] = decoder_attention
        
        # get the tokenized labels
        label_features = [{"input_ids": feature["labels"][0]} for feature in features]
        # pad the labels to max length
        label_batch = self.processor.tokenizer.pad(  # tokenizer pad since text
            label_features,
            padding=self.padding,
            return_tensors="np")    
        # replace padding with -100 to ignore loss correctly
        #decoder_inputs = decoder_input_batch["input_ids"].masked_fill(decoder_input_batch.attention_mask.ne(1), -100)
        label_inputs = label_batch["input_ids"]
        label_attention = label_batch["attention_mask"]
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (label_inputs[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            label_inputs = label_inputs[:, 1:]
        batch["labels"] = label_inputs
        batch["label_attention_mask"] = label_attention

        return batch
    

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


# in Flax, for seq2seq models we need to pass `decoder_input_ids`
# as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
# `shift_tokens_right` function
# copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids
    


def train(args):
    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    # model
    # FlaxWhisperForConditionalGeneration uses the FlaxWhisperPreTrainedModel forward method,
    # overrides the __call__ special method
    # FlaxWhisperForConditionalGeneration -> module_class = FlaxWhisperForConditionalGenerationModule
    # FlaxWhisperForConditionalGeneration(FlaxWhisperPreTrainedModel)
    # FlaxWhisperPreTrainedModel -> module = self.module_class
    # FlaxWhisperPreTrainedModel -> __call__ -> self.module.apply
    # FlaxWhisperForConditionalGenerationModule -> __call__ -> self.model -> FlaxWhisperModule
    model = FlaxWhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        seed=args.seed,
        dtype=getattr(jnp, args.dtype)
    )
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    model.config.suppress_tokens = []
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # save config maybe?

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

    # select small dataset for testing
    if args.max_train_samples is not None:
        common_voice["train"] = common_voice["train"].select(range(args.max_train_samples))

    if args.max_test_samples is not None:
        common_voice["test"] = common_voice["test"].select(range(args.max_test_samples))

    # resample to 16kHz
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=args.sampling_rate))


    # filter audio by duration 
    #max_input_length = args.max_duration * feature_extractor.sampling_rate
    #min_input_length = args.min_duration * feature_extractor.sampling_rate

    #def is_audio_in_length_range(length):
        #return length > min_input_length and length < max_input_length

    #common_voice = common_voice.filter(
        #is_audio_in_length_range,
        #num_proc=args.num_workers,
        #input_columns=["input_length"],
    #)

    # function to vectorize dataset
    # flax models need decoder_input_ids instead of labels
    # we need fixed length inputs for jitted functions
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        labels = tokenizer(batch["sentence"], return_tensors="np")
        # labels to compute loss
        batch["labels"] = labels.input_ids
        decoder_input_ids = shift_tokens_right(
            labels["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id
        )
        # decoder_input_ids to feed into the flax model
        # we will compute attention mask in data collator
        batch["decoder_input_ids"] = np.asarray(decoder_input_ids)

        return batch

    # vectorize dataset
    # input_features, decoder_input_ids, decoder_attention_mask, labels
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        desc="vectorize dataset"
    ) #, num_proc=2)



    # cer metric
    metric = evaluate.load("cer")

    # enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(log_dir=Path(args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )


    # initialize training
    rng = jax.random.PRNGKey(args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # compute effective batch size
    train_batch_size = int(args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(args.per_device_eval_batch_size) * jax.device_count()

    # create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=args.learning_rate, transition_steps=args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=args.learning_rate,
        end_value=0,
        transition_steps=args.train_steps - args.warmup_steps,
    )
    linear_decay_lr_schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[args.warmup_steps]
    )

    # we use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # the mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)
    
    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
        mask=decay_mask_fn,
    )

    # setup train state
    # FlaxWhisperForConditionalGenerationModule -> __call__ -> self.model -> FlaxWhisperModule
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)


    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum()
        # what is num_labels?
        num_labels = padding_mask.sum()
        return loss, num_labels
    

    # Define gradient update step fn
    # batch -> input_features, decoder_input_ids, decoder_attention_mask, labels, label_attention_mask
    def train_step(state, batch, label_smoothing_factor=0.0):
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            labels = batch.pop("labels")
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
            return loss, num_labels

        # value_and_grad
        # creates a function that evaluates both fun and the gradient of fun
        # returns a function with the same arguments as fun that evaluates both fun 
        # and the gradient of fun and returns them as a pair
        # argnums -> which positional argument(s) to differentiate with respect to (default 0).
        # if has_aux is True then a tuple of ((value, auxiliary_data), gradient) is returned.
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (loss, num_labels), grad = grad_fn(state.params)
        num_labels = jax.lax.psum(num_labels, "batch")

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        # true grad = total grad / total samples
        grad = jax.lax.psum(grad, "batch")
        grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)

        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        return new_state, metrics


    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]

        loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
        num_labels = jax.lax.psum(num_labels, "batch")

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        metrics = {"loss": loss}
        return metrics
    

    # Define generation function
    max_length = (
        args.generation_max_length if args.generation_max_length is not None else model.config.max_length
    )
    num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(
            batch["input_ids"],
            task=args.task,
            language=args.model_lang,
            is_multilingual=True,
            **gen_kwargs
        )
        return output_ids.sequences

    # create parallel version of the train and eval step
    # applying pmap() to a function will compile the function with XLA (similarly to jit()),
    # then execute it in parallel on XLA devices, such as multiple GPUs or multiple TPU cores.
    # it eplicates the function and executes each replica on its own XLA device in parallel.
    # donate_argnums -> specify which positional argument buffers are “donated” to the computation.
    # it is safe to donate argument buffers if you no longer need them once the computation has finished.
    # you should not reuse buffers that you donate to a computation,
    # jax will raise an error if you try to.
    # donate_argnums only work for positional arguments.
    p_train_step = jax.pmap(
        partial(train_step, label_smoothing_factor=args.label_smoothing_factor), "batch", donate_argnums=(0,)
    )
    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=args.label_smoothing_factor), "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # replicate the train state on each device
    state = state.replicate()


    # data collator # fix 
    data_collator = FlaxDataCollatorForWhisperFinetuning(processor=processor)

    # data loaders
    train_dataloader = DataLoader(
        common_voice["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )
    eval_dataloader = DataLoader(
        common_voice["test"],
        collate_fn=data_collator,
        batch_size=eval_batch_size,
    )


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(common_voice['train'])}")
    logger.info(f"  Num steps = {args.train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")

    ## check these ##
    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval

    # load from checkpoint (flax -> orbax)
    # add loading from checkpoint code here #


    # Training

    # main progress bar
    progress_bar = tqdm(range(global_step, args.train_steps), position=0)

    while True:
        # training time
        train_start = time.time()
        # train metrics
        train_metrics = []

        model.train()

        for batch in train_dataloader:
            batch = shard(batch)

            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)

            progress_bar.update(1)


            if (global_step + 1) % args.eval_steps == 0:
                train_time += time.time() - train_start

                train_metric = unreplicate(train_metric)

                progress_bar.write(
                    f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
                    f" {train_metric['learning_rate']})"
                )

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
                    predictions = processor.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    # we do not want to group tokens when computing the metrics
                    references = processor.batch_decode(references, group_tokens=False, skip_special_tokens=True, clean_up_tokenization_spaces=True)
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
        help="whether to skip steps already ccompleted while loading from checkpoint"
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
        "--per_device_train_batch_size",
        default=4, # 4
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
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
        '--num_workers',
        type=int,
        default=None, # None
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
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
        "--adam_epsilon",
        default=1e-6,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "--dtype",
        default='float16',
        type=str,
        help="Floating-point format in which the model weights should be initialized and trained. Choose one of [float32, float16, bfloat16]"
    )
    parser.add_argument(
        '--generation_max_length',
        type=int,
        default=225
    )
    parser.add_argument(
        '--label_smoothing_factor',
        type=float,
        default=0.0
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
            f"pass in dataset"
        )
    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'_jax_'+data_str
    print('output directory set to : {}'.format(args.output_dir))
    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )

    # setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training parameters {args}")
    
    # train function
    train(args)

            


if __name__ == "__main__":

    main()

