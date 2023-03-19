from dataclasses import dataclass
import datasets
from datasets import load_from_disk, DatasetDict
import re, json
from typing import Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoFeatureExtractor, 
    Wav2Vec2Processor,
    AutoProcessor,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from transformers.modeling_outputs import CausalLMOutput


_HIDDEN_STATES_START_POSITION = 2



def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: Optional[str] = None,
    unk_token: Optional[str] = None,
    pad_token: Optional[str] = None,
):
    # Given training and test labels create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets.column_names,
    )


    # take union of all unique characters in each dataset
    vocab_list = list(set(vocabs["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    # replace white space with delimiter token
    if word_delimiter_token is not None:
        vocab_dict[word_delimiter_token] = vocab_dict[" "]
        del vocab_dict[" "]

    # add unk and pad token
    if unk_token is not None:
        vocab_dict[unk_token] = len(vocab_dict)

    if pad_token is not None:
        vocab_dict[pad_token] = len(vocab_dict)

    return vocab_dict


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch
    


class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()


    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()


    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)


            print(flattened_targets)
            print(input_lengths)
            print(target_lengths)
            print(flattened_targets.shape)
            print(self.config.pad_token_id)

            # ctc_loss doesn't support fp16
            # input_size x batch_size x vocab_size
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            print(log_probs.shape)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    



def main():

    max_train_samples = 100
    audio_column = 'audio'
    text_column = 'sentence'
    max_duration = 20.0
    min_duration = 1.0
    model_name = 'facebook/wav2vec2-xls-r-300m'
    batch_size = 4
    chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    word_delimiter_token = "|"
    num_workers = 4
   

    # load cv data
    dataset = load_from_disk('/users/ujan/Downloads/common_voice_11')
    #dataset = load_from_disk('/home/ujan/Downloads/common_voice_11')
    dataset = dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    
    # sample data
    dataset = dataset["train"].select(range(max_train_samples))

    #print(dataset)

    def remove_special_characters(batch):
        if chars_to_ignore_regex is not None:
            batch["target_text"] = re.sub(chars_to_ignore_regex, "", batch[text_column]).lower() + " "
        else:
            batch["target_text"] = batch[text_column].lower() + " "
        return batch

    dataset = dataset.map(
        remove_special_characters,
        remove_columns=[text_column],
        desc="remove special characters from datasets",
    )

    # save special tokens for tokenizer
    word_delimiter_token = word_delimiter_token
    unk_token = unk_token
    pad_token = pad_token

    config = AutoConfig.from_pretrained(model_name)

    vocab_dict = create_vocabulary_from_data(
            dataset,
            word_delimiter_token=word_delimiter_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )
    
    with open('vocab.json', "w") as file:
        json.dump(vocab_dict, file)


    tokenizer_kwargs = {
        "config": config if config.tokenizer_class is not None else None,
        "tokenizer_type": config.model_type if config.tokenizer_class is None else None,
        "unk_token": unk_token,
        "pad_token": pad_token,
        "word_delimiter_token": word_delimiter_token,
    }

    tokenizer = AutoTokenizer.from_pretrained('./', **tokenizer_kwargs)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    dataset = dataset.cast_column(
        audio_column, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    # derive max & min input length for sample rate & max duration
    max_input_length = max_duration * feature_extractor.sampling_rate
    min_input_length = min_duration * feature_extractor.sampling_rate

    # We need to read the audio files as arrays and tokenize the targets.
    def prepare_dataset(batch):
        # load audio
        sample = batch[audio_column]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # encode targets
        additional_kwargs = {}

        batch["labels"] = tokenizer(batch["target_text"], **additional_kwargs).input_ids
        return batch


    dataset = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names,
        num_proc=num_workers,
        desc="preprocess dataset",
    )

    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    # filter data that is shorter than min_input_length
    print('filtering')
    dataset = dataset.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["input_length"],
    )

    # adapt config
    config.update(
        {
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),  # custom vocab
        }
    )

    print(len(tokenizer))

    # load model
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True)

    # Freeze Encoder 
    model.freeze_feature_encoder()

    # processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    batch = next(iter(dataloader))
    #print(batch['input_values'].shape)
    #print(batch['labels'].shape)
    #quit()

    outputs = model(**batch)



if __name__ == "__main__":
    main()