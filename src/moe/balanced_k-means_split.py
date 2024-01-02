"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

from os.path import dirname, abspath
from tqdm.auto import tqdm

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
import torch
from torch import nn
from typing import List
from modeling_whisper_moe import WhisperForConditionalGeneration

from transformers import set_seed
import argparse

from k_means_constrained import KMeansConstrained

#torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=50000))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)
    

def get_permutation_matrix(labels, ffn_dim, num_experts: int, expert_size: int):

    P = torch.zeros(ffn_dim, ffn_dim)
    for i in range(num_experts):
        e_ids = (labels==i).nonzero()[0].tolist()  # 0 -> ffn_dim
        for j in range(expert_size):
            P[i*expert_size + j, e_ids[j]] = 1

    return P

def cluster(model, k_means, num_experts: int, expert_size: int, encoder=True):

    config = model.config

    if encoder:
        num_layers = config.encoder_layers
        ffn_dim = config.encoder_ffn_dim
    else:
        num_layers = config.decoder_layers
        ffn_dim = config.decoder_ffn_dim

    layer_bar = tqdm(range(num_layers))
    for l in range(num_layers):
        # encoder
        if encoder:
            W1 = model.model.encoder.layers[l].fc1.weight
            b1 = model.model.encoder.layers[l].fc1.bias
            W2 = model.model.encoder.layers[l].fc2.weight
            k_means.fit_predict(W1)
            labels = k_means.labels_
            # permutation_matrix
            P = get_permutation_matrix(labels, ffn_dim, num_experts, expert_size)
            # permute
            with torch.no_grad():
                # permute and set W1, b1
                model.model.encoder.layers[l].fc1.weight = nn.Parameter(P@W1)
                model.model.encoder.layers[l].fc1.bias = nn.Parameter(P@b1)
                # permute and set W2
                PT = torch.transpose(P,0,1)
                model.model.encoder.layers[l].fc2.weight = nn.Parameter(W2@PT)
        # decoder
        else:
            W1 = model.model.decoder.layers[l].fc1.weight
            b1 = model.model.decoder.layers[l].fc1.bias
            W2 = model.model.decoder.layers[l].fc2.weight
            k_means.fit_predict(W1)
            labels = k_means.labels_
            # permutation_matrix
            P = get_permutation_matrix(labels, ffn_dim, num_experts, expert_size)
            # permute
            with torch.no_grad():
                # permute and set W1, b1
                model.model.decoder.layers[l].fc1.weight = nn.Parameter(P@W1)
                model.model.decoder.layers[l].fc1.bias = nn.Parameter(P@b1)
                # permute and set W2
                PT = torch.transpose(P,0,1)
                model.model.decoder.layers[l].fc2.weight = nn.Parameter(W2@PT)
                
        layer_bar.update(1)

    return model



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
        default='relu',
        type=str,
        help="change model activation function",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
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
        "--mixed_precision",
        default='fp16',
        type=str,
    )
    parser.add_argument(
        "--num_experts",
        default=96,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
    )

    # parse args
    args = parser.parse_args()
    args.eval_batch_size = 1

    # set seed
    set_seed(args.seed)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )
    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'_moe_k_means'
    print('output directory set to : {}'.format(args.output_dir))

    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    
    # model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        #config=config,
        #cache_dir=args.cache_dir if args.cache_dir else None,
    )
    #model.config.forced_decoder_ids = None
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    model.config.suppress_tokens = []

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    if args.activation is not None:
        model.config.activation_function = args.activation
        print('activation changed to {}'.format(model.config.activation_function))

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    expert_size = model.config.encoder_ffn_dim // args.num_experts 
    k_means = KMeansConstrained(
        n_clusters=args.num_experts,
        size_min=expert_size,
        size_max=expert_size,
        random_state=0
    )

    with torch.no_grad():
        print('clustering encoder')
        model = cluster(model, k_means, args.num_experts, expert_size, encoder=True)
        print('clustering decoder')
        model = cluster(model, k_means, args.num_experts, expert_size, encoder=False)

    # update config with num experts
    model.config.update({'num_experts': args.num_experts})

    # save model
    print('saving moefied model')
    feature_extractor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)

      


if __name__ == "__main__":

    run()


