import math
import copy
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
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


import datasets
from datasets import load_from_disk
from typing import Dict, List, Optional, Union
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW



@dataclass
class Data2VecAudioForPreTrainingOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    #hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    #attentions: Optional[Tuple[torch.FloatTensor]] = None




class Data2VecAudioForPreTraining(Data2VecAudioPreTrainedModel):

    # check mask time, mask feature

    def __init__(self, config: Data2VecAudioConfig, top_k, tau, tau_e, n_tau, skip_keys=None):
        super().__init__(config)
        # model in student mode
        self.student = Data2VecAudioModel(config) 
        # model in teacher mode
        self.teacher = copy.deepcopy(self.student)
        # keys to skip during teacher update (maybe teacher already trained)
        self.skip_keys = skip_keys
        # hidden dimension for regression head
        self.embed_dim = self.config.hidden_size
        # where to set k and tau?
        self.top_k = top_k # 8, num top layers to average representations for target
        # initial decay
        self.tau = tau  # 0.90
        # final decay
        self.tau_e = tau_e  # 0.99
        # number of steps to reach final from initial
        self.n_tau = n_tau  # 1000
        # current steps
        self.c_tau = 0
        # affine = True -> learnable paramters
        self.instance_norm = nn.InstanceNorm1d(self.embed_dim, affine=True)
        # regression head
        self.regression_head = nn.Linear(self.embed_dim, self.embed_dim)
        # loss
        self.mse_loss = nn.MSELoss()



    # check device
    def ema(self):

        # get current decay
        # if number of steps exceeded n_tau, set tau = tau_e
        if self.tau != self.tau_e:
            if self.c_tau >= self.n_tau:
                self.tau = self.tau_e
        # else linearly increase tau 
        else:
            r = self.tau_e - self.tau
            pct_remaining = 1 - self.c_tau / self.n_tau
            self.tau = self.tau_e - r * pct_remaining
            self.c_tau += 1
        
        # update teacher weights
        if self.tau < 1:
            ema_state_dict = {}
            ema_params = self.teacher.state_dict()
            for key, param in self.student.state_dict().items():
                ema_param = ema_params[key].float()
                if self.skip_keys is not None and key in self.skip_keys:
                    ema_param = param.to(dtype=ema_param.dtype).clone()
                else:
                    # mul_, add_ inplace versions
                    ema_param.mul_(self.tau)
                    ema_param.add_(param.to(dtype=ema_param.dtype), alpha=1 - self.tau)
                ema_state_dict[key] = ema_param
            self.teacher.load_state_dict(ema_state_dict)  # strict=False

        else:
            raise ValueError(
                f"tau exceeding 1"
            )



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
        output_attentions: Optional[bool] = None,  # whether to output attentions from pretraining model
        output_hidden_states: Optional[bool] = None,  # whether to output hidden states from pretraining model
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Data2VecAudioForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)

        # pass inputs though model in student mode with mask
        # outputs[0] -> masked (spec augment), last, projected, hidden state, hidden_state dim (768)
        # outputs[1] -> unmasked, un-projected, conv feature, layer norm, xvector dim (512)
        outputs = self.student(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            mask_time_indices=mask_time_indices,
            return_dict=return_dict,
        )
        # encoder outputs with masked conv features after layer norm and projection

        masked_final_state = outputs[0] # [b,s,d], predict unmasked from this
        # pass x though regression head
        masked_final_state = self.regression_head(masked_final_state)  # X (prediction)

        # pass inputs though model in teacher mode without mask
        # outputs[0] -> unmasked (check mask_time_indices None), last, projected, hidden state, hidden_state dim (768)
        # outputs[1] -> unmasked, un-projected, conv feature, layer norm, xvector dim (512)
        # outputs[2] -> unmasked, all hidden states (768)
        outputs = self.teacher(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # need top k unmasked hidden states for target
            mask_time_indices=None,  # no mask
            return_dict=return_dict,
        )
        unmasked_hidden_states = outputs[2]  # [b,s,d] x 13

        # take top k layer representations
        unmasked_hidden_states = unmasked_hidden_states[-self.top_k:]     
        # instance normalization of target for each block (layer)
        # transpose to d x s and then back after instance norm
        unmasked_hidden_states = [self.instance_norm(l.transpose(1, 2)).transpose(1,2) for l in unmasked_hidden_states]
        # average top k blocks after instance norm
        unmasked_k_avg = torch.mean(torch.stack(unmasked_hidden_states), dim=0)  # Y (target)

        # l2(MSE) loss
        loss = self.mse_loss(masked_final_state, unmasked_k_avg)
    

        return Data2VecAudioForPreTrainingOutput(loss=loss)

        



@dataclass
class DataCollatorForData2VecPretraining:
    """
    Data collator that will dynamically pad the inputs received and prepare masked indices
    for self-supervised pretraining.

    Args:
        model (:class:`~transformers.Wav2Vec2ForPreTraining`):
            The Wav2Vec2 model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
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
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: Data2VecAudioForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor  # same for data2vec
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        mask_indices_seq_length = self.model._get_feat_extract_output_lengths(batch["input_values"].shape[-1])
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        features_shape = (batch_size, mask_indices_seq_length)

        # sample randomly masked indices for spec augment
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        #sampled_negative_indices = _sample_negative_indices(
            #features_shape,
            #self.model.config.num_negatives,
            #mask_time_indices=mask_time_indices,
        #)
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        #batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch



## demo to test classes ##
def main():

    max_train_samples = 100
    audio_column_name = 'audio'
    max_duration_in_seconds = 20.0
    min_duration_in_seconds = 1.0
    model_name = 'facebook/data2vec-audio-base'
    batch_size = 4
   

    # load cv data
    dataset = load_from_disk('/users/ujan/Downloads/common_voice_11')
    #dataset = load_from_disk('/home/ujan/Downloads/common_voice_11')
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


    #print(dataset)

    # config 
    config = Data2VecAudioConfig.from_pretrained(model_name)
    # model
    model = Data2VecAudioForPreTraining(config, 8, 0.9, 0.99, 1000)

    mask_time_prob = config.mask_time_prob 
    mask_time_length = config.mask_time_length 

    # data collator, data loader
    data_collator = DataCollatorForData2VecPretraining(
        model=model,
        feature_extractor=feature_extractor,
        #pad_to_multiple_of=args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    for name, param in model.named_parameters():
        if 'teacher' in name:
            print(name, param.grad)
            break

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)

    # get a batch
    batch = next(iter(dataloader))
    #print(batch['input_values'].shape)
    #quit()

    loss = model(**batch).loss
    
    loss.backward()

    # do not update teacher. only student. teacher updated through ema
    model.teacher.zero_grad()

    optimizer.step()
    optimizer.zero_grad()

    # update teacher weights
    model.ema()

    print('done')




if __name__ == "__main__":

    main()