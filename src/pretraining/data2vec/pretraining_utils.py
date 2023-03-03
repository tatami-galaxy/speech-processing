import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    Data2VecAudioPreTrainedModel,
    Data2VecAudioConfig,
    Data2VecAudioModel
)

from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import torch_int_div
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging


logger = logging.get_logger(__name__)


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
    ) -> Union[Tuple, Wav2Vec2ForPreTrainingOutput]:

    


def main():
    pass



if __name__ == "__main__":

    main()