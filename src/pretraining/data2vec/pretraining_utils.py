import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from dataclasses import dataclass

from transformers import (
    Data2VecAudioPreTrainedModel,
    Data2VecAudioConfig,
    Data2VecAudioModel,
    Wav2Vec2FeatureExtractor
)

from transformers.utils.generic import ModelOutput


import datasets
from datasets import load_from_disk



@dataclass
class Data2VecAudioForPreTrainingOutput(ModelOutput):
    pass




class Data2VecAudioForPreTraining(Data2VecAudioPreTrainedModel):

    def __init__(self, config: Data2VecAudioConfig):
        super().__init__(config)
        self.data2vec = Data2VecAudioModel(config)


    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.data2vec.feature_extractor._freeze_parameters()


    #@replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    #@add_start_docstrings_to_model_forward(DATA2VEC_AUDIO_INPUTS_DOCSTRING)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Data2VecAudioForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )

        # check outputs
        print(outputs)

    

## demo to test classes ##
def main():

    max_train_samples = 100
    audio_column_name = 'audio'
    max_duration_in_seconds = 20.0
    min_duration_in_seconds = 1.0
   

    # load cv data
    dataset = load_from_disk('/users/ujan/Downloads/common_voice_11')
    dataset = dataset.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
    
    # sample data
    dataset = dataset["train"].select(range(max_train_samples))

    feature_extractor = Wav2Vec2FeatureExtractor()
    dataset = dataset.cast_column(
        audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    max_length = int(max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(min_duration_in_seconds * feature_extractor.sampling_rate)
    
    def prepare_dataset(batch):
        sample = batch[audio_column_name]

        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], max_length=max_length, truncation=True
        )
        batch["input_values"] = inputs.input_values[0]
        batch["input_length"] = len(inputs.input_values[0])

        return batch
    
    dataset = dataset.map(
        prepare_dataset,
        num_proc=2,
        remove_columns=dataset.column_names,
    )

    if min_length > 0.0:
        dataset = dataset.filter(
            lambda x: x > min_length,
            num_proc=2,
            input_columns=["input_length"],
        )

        dataset = dataset.remove_columns("input_length")


    print(dataset)

    # config 

    # model

    # data collator

    # get a batch




if __name__ == "__main__":

    main()