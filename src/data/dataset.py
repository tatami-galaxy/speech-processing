import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets

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
    
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if args.huggingface:
        dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
        #print(dataset)
        


if __name__ == "__main__":
    main()
