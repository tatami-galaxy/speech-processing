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
from tqdm.auto import tqdm
import time, math
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate, pad_shard_unpad
import orbax
from flax.training import (
    train_state,
    orbax_utils
)
from flax.training.common_utils import (
    onehot,
    shard,
    get_metrics
)

import transformers
from transformers import (
    GenerationConfig,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    FlaxWhisperForConditionalGeneration,
)
from transformers import (
    set_seed,
    is_tensorboard_available,
)
from transformers.utils import send_example_telemetry

import datasets
from datasets import (
    Dataset,
    load_dataset,
    DatasetDict,
    Audio
)

import evaluate

from multiprocess import set_start_method


#jax.config.update('jax_array', False) -> only works below jax and jaxlib 0.4.6
logger = logging.getLogger(__name__)

# sending telemetry
# tracking the example usage helps us better allocate resources to maintain them
# the information sent is the one passed as arguments along with your Python/PyTorch versions
send_example_telemetry("run_summarization", framework="flax")


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)

# constants
LANG_TO_ID = {"hindi" : "<|hi|>"}
LANG_TO_ID = {"chinese" : "<|zh|>"}

    

#class TrainState(train_state.TrainState):
    #dropout_rng: jnp.ndarray

    #def replicate(self):
        #return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng.reshape(-1)))
    

def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch
    


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


def eval(args):
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
    # input_shape: typing.Tuple[int] = (b, 80, 3000)
    model = FlaxWhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        seed=args.seed,
        dtype=getattr(jnp, args.dtype),
        from_pt=args.from_pt
    )
    #model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    #model.config.suppress_tokens = []

    #model.config.suppress_tokens.extend(tokenizer.encode('a', add_special_tokens=False))
    #print("model config supress tokens")
    #print(model.config.suppress_tokens)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")  
    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False


    #print('config supress tokens :')
    #print(model.config.suppress_tokens)

    # save config maybe?

    # dataset
    common_voice = DatasetDict()
    #common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test", use_auth_token=True)

    # remove unused columns
    common_voice = common_voice.remove_columns(
        [
            "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
        ]
    )

    # select small dataset for testing
    #if args.max_train_samples is not None:
        #common_voice["train"] = common_voice["train"].select(range(args.max_train_samples))

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

    # tokenizer and generation max length
    max_length = (
        args.generation_max_length if args.generation_max_length is not None else model.config.max_length
    )

    # function to vectorize dataset
    # flax models need decoder_input_ids instead of labels
    # we need fixed length inputs for jitted functions
    # https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/whisper/feature_extraction_whisper.py#L254
    #if return_attention_mask:
        # rescale from sample (48000) to feature (3000)
    def prepare_dataset(batch):  #, rank):

        #os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % jax.device_count())

        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        # 80 x 3000
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        labels = tokenizer(
            batch["sentence"],
            padding="max_length",
            max_length=max_length,
            return_tensors="np")

        # labels to compute loss
        # 1 x generation length or max length
        batch["labels"] = labels["input_ids"].flatten()
        decoder_input_ids = shift_tokens_right(
            labels["input_ids"], model.config.pad_token_id, model.config.decoder_start_token_id
        )
        # decoder_input_ids to feed into the flax model
        batch["decoder_input_ids"] = np.asarray(decoder_input_ids).flatten()

        # we need decoder_attention_mask so we can ignore pad tokens from loss
        # completely masks decoder_input_ids
        # leaves first pad token (after input ids) unmasked in labels
        # need different mask for labels?
        batch["decoder_attention_mask"] = labels["attention_mask"].flatten()

        return batch

    # vectorize dataset
    # input_features, decoder_input_ids, decoder_attention_mask, labels
    common_voice = common_voice.map(
        prepare_dataset,
        #with_rank=True,
        remove_columns=common_voice.column_names["test"],
        desc="vectorize dataset",
        num_proc=args.num_workers,
    ) 

    # test dataset
    #train_dataset = common_voice["train"]
    test_dataset = common_voice["test"]


    # metrics
    cer = evaluate.load("cer")
    wer = evaluate.load("wer")

    def compute_metrics(preds, labels):
        result = {}
        predictions = processor.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        references = processor.batch_decode(
            labels,
            group_tokens=False,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        # compute cer, wer
        result["cer"] = cer.compute(predictions=predictions, references=references)
        result["wer"] = wer.compute(predictions=predictions, references=references)
        return result
        

    # enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(log_dir=Path(args.checkpoint_dir))
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


    # compute effective batch size
    #train_batch_size = int(args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(args.per_device_eval_batch_size) * jax.device_count()

    # eval steps in eval_dataset
    # different from args.eval_steps
    eval_steps = math.ceil(len(common_voice["test"]) / eval_batch_size)

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

    # data collator
    #data_collator = FlaxDataCollatorForWhisperFinetuning(processor=processor)

    # data loaders
    #train_loader = DataLoader(
        #common_voice["train"],
        #shuffle=True,
        #collate_fn=data_collator,
        #batch_size=train_batch_size,
    #)
    #eval_loader = DataLoader(
        #common_voice["test"],
        #collate_fn=data_collator,
        #batch_size=eval_batch_size,
    #)

    # setup train state
    # FlaxWhisperForConditionalGenerationModule -> __call__ -> self.model -> FlaxWhisperModule
    #state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=adamw)


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
    

    # define gradient update step fn
    # batch -> input_features, decoder_input_ids, decoder_attention_mask, labels
    # cant print values inside a jit compiled function

    # pmap -> replicate your model on devices, shard your data,
    # and have each calculate their individual loss and gradients.
    # pmean to average them across all devices and apply your gradient (psum)

    # define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):

        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]

        loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
        num_labels = jax.lax.psum(num_labels, "batch") # AllReduce

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")  # AllReduce
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        metrics = {"loss": loss}
        return metrics
    

    # generation functions

    def make_generation_config(supress_en=False):

        generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
        gen_dict = generation_config.to_dict()
        # add attributes to genration_config
        # generation_config does not have "langauge", but generate() tries to use it
        # can be empty dict here since being set in generate_step
        gen_dict["language"] = LANG_TO_ID[args.model_lang]
        if supress_en:
            # en tokens to suppress from multilingual vocab
            en_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny.en")  # change if loaded locally
            suppress_en_list = []
            for key in en_tokenizer.encoder.keys():
                if key in tokenizer.encoder.keys() and key.isalpha():
                    suppress_en_list.append(key)
            # supress english tokens
            gen_dict['suppress_tokens'].extend(tokenizer.encode(suppress_en_list, add_special_tokens=False))

        # reload with new attributes
        generation_config = GenerationConfig.from_dict(gen_dict)

        return generation_config


    # max_length defined after tokenizer
    num_beams = args.num_beams if args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    # generation config
    generation_config = make_generation_config(supress_en=False)

    # batch -> input_features, decoder_input_ids, decoder_attention_mask, labels
    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(
            batch["input_features"],
            generation_config=generation_config,
            task=args.task,
            language=LANG_TO_ID[args.model_lang],  # set lang here
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
    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=args.label_smoothing_factor), "batch")
    p_generate_step = jax.pmap(generate_step, "batch")


    if args.checkpoint_dir is not None:

        print("loading from : {}".format(args.checkpoint_dir))

        # init checkpointer
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=args.max_to_keep, create=True)
        # checkpoint manager
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            args.checkpoint_dir,
            orbax_checkpointer,
            options
    )     

        # load from previous checkpoint
        # get latest checkpoint
        step = checkpoint_manager.latest_step()  # or choose step
        print('restoring step : {}'.format(step))

        # empty state and config to load state into
        empty_state = train_state.TrainState.create(
            apply_fn=model.__call__,
            params=jax.tree_map(jnp.zeros_like, model.params),  # values of the tree leaf doesn't matter
            tx=adamw,
            #dropout_rng=dropout_rng
        )
        empty_config = model.config
        #target = {'model': empty_state, 'config': empty_config, 'data': [jnp.zeros_like(x1)]}
        target = {'state': empty_state, 'config': empty_config}  # state or model -> automate maybe

        # restore
        restored = checkpoint_manager.restore(step, items=target)
        state = restored['state']



    # write fixed hyoerparameters to tensorboard
    if has_tensorboard and jax.process_index() == 0:
        summary_writer.scalar("eval_batch_size", eval_batch_size, step)

    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous eval batch size per device = {args.per_device_eval_batch_size}")


    # replicate the train state on each device
    #state = state.replicate()
    state = jax_utils.replicate(state)

    # Training

    # initialize training
    rng = jax.random.PRNGKey(args.seed)

    eval_metrics = []
    eval_preds = []
    eval_labels = []
    result_dict = {}

    # main progress bar
    eval_bar = tqdm(range(eval_steps), position=0)

    # create sampling rng
    rng, input_rng = jax.random.split(rng)

    eval_loader = data_loader(input_rng, test_dataset, eval_batch_size, shuffle=True)

    for batch in eval_loader:
        labels = batch["labels"]
        metrics = pad_shard_unpad(p_eval_step, static_return=True)(
            state.params, batch, min_device_batch=args.per_device_eval_batch_size)
        eval_metrics.append(metrics)

        # generation
        generated_ids = pad_shard_unpad(p_generate_step)(state.params, batch)
        eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
        eval_labels.extend(labels)
    
        eval_bar.update(1)

    # normalize eval metrics
    # eval metrics (loss)
    eval_metrics = get_metrics(eval_metrics)  # dict
    eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)  # dict
    # cer, wer
    result = compute_metrics(eval_preds, eval_labels)
    eval_metrics.update(result)

    # collect results together
    result_dict['eval_loss'] = eval_metrics['loss']
    result_dict['cer'] = eval_metrics['cer']
    result_dict['wer'] = eval_metrics['wer']

    # write to terminal and tensorboard
    for key, val in result_dict.items():
        print('{} : {}'.format(key, val))
    if has_tensorboard and jax.process_index() == 0:
        for key, val in result_dict.items():
            summary_writer.scalar(key, val, step)



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
        "--from_pt",
        action="store_true",
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
        "--checkpoint_dir",
        default=None,
        type=str,
        help="Has to be folder containing step folder. Not step folder itself",
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
        '--max_test_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=4,
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
        "--eval_steps",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--max_to_keep",
        default=3,
        type=int,
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=os.cpu_count(), # os.cpu_count()
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
    if args.checkpoint_dir is None:
        print('checkpoint None -> evalulating base model for {}'.format(args.model_name_or_path))
    else:
        print('checkpoint directory set to : {}'.format(args.checkpoint_dir))

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
    eval(args)

            


if __name__ == "__main__":
    #set_start_method("spawn")
    main()

