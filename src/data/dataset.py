import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The name of the dataset to use.",
    )
    parser.add_argument(
        "--huggingface",
        type=bool,
        default=True,
        help="If its on HuggingFace",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="Cache directory",
    )

def main():


    if args.huggingface:
        dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)