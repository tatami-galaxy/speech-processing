import os
import subprocess
import argparse


def main():

    torch_req = ['torch', 'torchvision', 'torchaudio', 'accelerate']
    jax_req = ['gdown', 'flax']
    req = ['transformers', 'datasets', 'evaluate', 'jiwer', 'librosa', 'soundfile', 'tensorboard', 'gdown']

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--torch",
        action="store_true",
        help="install packages to run torch code"
    )
    parser.add_argument(
        "--jax",
        action="store_true",
        help="install packages to run jax/flax code"
    )

    # parse args
    args = parser.parse_args()

    # directorries
    os.mkdir('data')
    os.mkdir('data/raw')
    os.mkdir('data/interim')
    os.mkdir('data/processed')

    os.mkdir('models')
    os.mkdir('models/whisper')
    os.mkdir('models/wav2vec2')
    os.mkdir('models/data2vec')
    os.mkdir('models/pretrained_models')
    os.mkdir('models/pretrained_models/cont_pre_training')

    # packages
    if args.torch:
        req[:0] = torch_req
    elif args.jax:
        req.extend(jax_req)

    # install packages
    for package in req:
        subprocess.run(["pip", "install", package]) 

    # huggingface-cli login 
    # accelerate config


if __name__ == "__main__":
    main()

 
