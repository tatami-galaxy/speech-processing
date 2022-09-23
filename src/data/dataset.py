import argparse
import datasets
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import Wav2Vec2FeatureExtractor, set_seed

def parse_args():

  parser = argparse.ArgumentParser(description="Preprocess dataset")

  parser.add_argument("--dataset", type=str, default=None,
    help="The name of the dataset to use.",
    required=True,)

  parser.add_argument("--huggingface", type=bool, default=True,
    help="If its on HuggingFace",)

  parser.add_argument("--cache_dir", type=str,default="~/.cache/huggingface/datasets",
    help="Cache directory",)

  parser.add_argument("--processed_data_dir", type=str,default="~/speech-processing/data/processed",
    help="Processed data directory.",)

  parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")

  parser.add_argument("--sampling_rate", type=int, default=16000,
    help="Audio sampling rate. Default=16000 for wav2vec2",)

  parser.add_argument("--audio_column_name", type=str, default="audio",
    help="Column in the dataset that contains speech file path. Defaults to 'audio'",)

  parser.add_argument("--max_duration_in_seconds", type=float, default=20.0,
    help="Filter out audio files that are longer than `max_duration_in_seconds` seconds",)

  parser.add_argument("--min_duration_in_seconds", type=float, default=2.0,
    help="Filter out audio files that are shorter than `min_duration_in_seconds` seconds",)

  parser.add_argument("--preprocessing_num_workers", type=int, default=None,
    help="The number of processes to use for the preprocessing.",)
    
  args = parser.parse_args()

  return args



def main():

  args = parse_args()

  if args.huggingface:
    # load dataset from huggingface or cache
    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)
    # concatenate train, validation amd test splits
    train_sets = []
    valid_sets = []
    test_sets = []
    for d_set in dataset:
      if 'train' in d_set: train_sets.append(dataset[d_set])
      elif 'validation' in d_set: valid_sets.append(dataset[d_set])
      elif 'test' in d_set: test_sets.append(dataset[d_set])
    trainset = concatenate_datasets(train_sets).shuffle(seed=args.seed)
    validset = concatenate_datasets(valid_sets).shuffle(seed=args.seed)
    testset = concatenate_datasets(test_sets).shuffle(seed=args.seed)
    raw_datasets = DatasetDict({'train':trainset, 'validation':validset, 'test':testset})

  # preprocessing

  # datasets takes care of automatically loading and resampling the audio,
  # so that we just need to set the correct target sampling rate and normalize the input
  # via feature_extractor 

  # wav2vec2 models that have set config.feat_extract_norm == "group", such as wav2vec2-base, have not been trained using attention_mask
  # for such models, input_values should simply be padded with 0 and no attention_mask should be passed.
  # for wav2vec2 models that have set config.feat_extract_norm == "layer", such as wav2vec2-lv60, attention_mask should be passed for batched inference.

  # huggingface pretraining script only supports "newer" stable layer norm architecture
  # apply_spec_augment has to be True, mask_feature_prob has to be 0.0


  feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=args.sampling_rate,
    padding_value=0.0, do_normalize=True, return_attention_mask=False)

  # make sure that dataset decodes audio with correct sampling rate
  raw_datasets = raw_datasets.cast_column(args.audio_column_name,
    datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate))
    
  # only normalized-inputs-training is supported
  if not feature_extractor.do_normalize:
    raise ValueError(
      "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
    )

  # set max & min audio length in number of samples
  max_length = int(args.max_duration_in_seconds * feature_extractor.sampling_rate)
  min_length = int(args.min_duration_in_seconds * feature_extractor.sampling_rate)

  # vectorization 

  def prepare_dataset(batch):

    sample = batch[args.audio_column_name]

    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"],
      max_length=max_length, truncation=True)

    batch["input_values"] = inputs.input_values[0]  # where is this used?
    batch["input_length"] = len(inputs.input_values[0])

    return batch


  vectorized_datasets = raw_datasets.map(prepare_dataset, 
    num_proc=args.preprocessing_num_workers, remove_columns=raw_datasets["train"].column_names,)

  if min_length > 0.0:
    vectorized_datasets = vectorized_datasets.filter(
      lambda x: x > min_length,
      num_proc=args.preprocessing_num_workers,
      input_columns=["input_length"],)


  vectorized_datasets = vectorized_datasets.remove_columns("input_length")

  # save to disk
  vectorized_datasets.save_to_disk(args.processed_data_dir)
     


if __name__ == "__main__":
  main()
