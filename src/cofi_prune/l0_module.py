import pdb

from re import L
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from transformers.utils import logging


limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
logger = logging.get_logger(__name__)


class L0Module(Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 magical_number=0.8, # from Wang et al. 2020
                 ):
        
        super(L0Module, self).__init__()

        self.types = [
            "hidden", 
            "head",
            "mha",
            "ffn_dim",
            "ffn",
        ]
        
        # model parameters
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model  # hidden_size
        self.encoder_layers = config.encoder_layers
        self.encoder_attention_heads = config.encoder_attention_heads  # num_attention_heads
        self.decoder_layers = config.decoder_layers
        self.decoder_attention_heads = config.decoder_attention_heads  # num_attention_heads
        self.decoder_ffn_dim = config.decoder_ffn_dim  # intermediate_size
        self.encoder_ffn_dim = config.encoder_ffn_dim  # intermediate_size
        self.num_hidden_layers = config.encoder_layers + config.decoder_layers  # encoder + decoder layers
        self.ffn_num_per_layer = 1  # linear -> activation -> linear
        self.dim_per_encoder_head = self.d_model // self.encoder_attention_heads
        self.dim_per_decoder_head = self.d_model // self.decoder_attention_heads

        # same number of heads, head size for encoder, decoder 
        # 4 self attention weight matrices. 4 weights, 3 biases
        # no bias for self_attn.k_proj
        self.params_per_head_layer = self.d_model * self.d_model * 4  + self.d_model * 3
        self.params_per_head =  self.params_per_head_layer // self.encoder_attention_heads
        
        # same intermediate size for encoder, decoder
        # weights, biases for each ffn layer
        self.params_per_ffn_layer = self.d_model * config.encoder_ffn_dim * 2  + self.d_model + config.encoder_ffn_dim 
        self.params_per_encoder_ffn_dim = self.params_per_ffn_layer // self.encoder_ffn_dim

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_ffn_layer) * self.num_hidden_layers
        self.prunable_model_size = 0 

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        # initialize all zs
        # "type" in initialize must be one of self.types
        self.initialize_hidden()
        self.initialize_head()
        self.initialize_mha()
        self.initialize_ffn_dim()
        self.initialize_ffn() 
        

        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity


    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup

    
    # init the z_logas
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape):
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self):
        # shape -> d_model
        self.hidden_loga = self.initialize_parameters(self.d_model)
        self.add_one_module(
            self.hidden_loga,
            type="hidden", 
            ## check ##
            # is parameters_per_dim being used anywhere?
            parameter_per_dim=self.d_model * 4 + self.d_model* 4 * 2,  # ??
            # how is size used?
            size=self.d_model, shape=[self.d_model]
        )
        self.reset_loga(self.hidden_loga, mean=10)  # different mean?
        #logger.info(f"Initialized hidden loga. Prunable_model_size = {self.prunable_model_size}")


    def initialize_head(self, add_prunable_model_size=True):
        self.head_loga = self.initialize_parameters(self.encoder_attention_heads, self.num_hidden_layers)
        self.reset_loga(self.head_loga, mean=10)
        self.add_one_module(self.head_loga, type="head", 
                            parameter_per_dim=self.params_per_head,
                            #  both encoder and decoder attentions heads
                            size=self.encoder_attention_heads+self.decoder_attention_heads,
                            # head size same across encoder, decoder
                            # how is size used?
                            shape=[self.num_hidden_layers, 1, self.encoder_attention_heads, 1, 1])
        if add_prunable_model_size:
            self.prunable_model_size += self.params_per_head * self.num_hidden_layers * (self.encoder_attention_heads + self.decoder_attention_heads)
        #logger.info(f"Initialized heads. Prunable_model_size = {self.prunable_model_size}")


    def initialize_mha(self):
        n_layer = self.num_hidden_layers
        self.headlayer_loga = self.initialize_parameters(n_layer)
        self.reset_loga(self.headlayer_loga, mean=10)
        self.add_one_module(self.headlayer_loga, type="mha", 
                            parameter_per_dim=self.params_per_head * (self.encoder_attention_heads + self.decoder_attention_heads), size=1,
                            shape=[n_layer])
        #logger.info(f"Initialized layerwise mha. Prunable_model_size = {self.prunable_model_size}")

    ## change to ffn dimensions ##
    def initialize_ffn_dim(self):
        self.int_loga = self.initialize_parameters(self.encoder_ffn_dim, self.num_hidden_layers)

        self.add_one_module(self.int_loga, type="ffn_dim", 
                            parameter_per_dim=self.params_per_encoder_ffn_dim, size=self.encoder_ffn_dim,
                            shape=[self.num_hidden_layers, 1, 1, self.encoder_ffn_dim])
        self.prunable_model_size += self.params_per_ffn_layer * self.num_hidden_layers
        self.reset_loga(self.int_loga)
        #logger.info(f"Initialized ffn dim. Prunable_model_size = {self.prunable_model_size}")

    ## change to ffn. dont drop entire layer ##
    def initialize_ffn(self):
        n_layer = self.num_hidden_layers
        self.intlayer_loga = self.initialize_parameters(n_layer)
        self.add_one_module(self.intlayer_loga, type="ffn", 
                            parameter_per_dim=self.params_per_ffn_layer, size=self.ffn_num_per_layer,
                            shape=[n_layer])
        self.reset_loga(self.intlayer_loga, mean=10)
        #logger.info(f"Initialized ffn. Prunable_model_size = {self.prunable_model_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)


    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "ffn", "head"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])


    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])


    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)


    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a


    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size


    def transform_scores_for_head(self):

        all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga)
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score


    def get_num_parameters_for_ffn(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters


    def get_num_parameters_and_constraint_for_hidden(self): #! calculate the current parsity
        num_parameters = 0
       
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.d_model

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2
        return num_parameters


    def get_num_parameters_and_constraint(self):
        num_parameters = 0

        all_head_score, head_score = self.transform_scores_for_head()
        
        head_score = head_score * all_head_score
        num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = int_score * intlayer_score
        num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):

        target_sparsity = self.target_sparsity
        expected_size = self.get_num_parameters_and_constraint_for_hidden() #! calculate \bar s
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = ( #! see appendix
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2 #! where is the lambda 1 and lambda 2 from
        )
        return lagrangian_loss, expected_sparsity, target_sparsity

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps


    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z


    # during inference
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask


    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.types:
            name = type[:-2]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                new_z = z.squeeze().detach().cpu().numpy() > 0
            numpified_zs[name] = new_z
        return numpified_zs


    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        hidden_z = numpified_zs["hidden"]
        intermediate_z = numpified_zs["intermediate"]
        ffn_z = numpified_zs["ffn"].reshape(-1, 1)
        head_z = numpified_zs["head"]
        head_layer_z = numpified_zs["head_layer"].reshape(-1, 1)

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.num_hidden_layers, self.encoder_ffn_dim).sum(-1).tolist()
        remaining_head_nums = head_z.reshape(self.num_hidden_layers, self.encoder_attention_heads).sum(-1).tolist()

        head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()
        intermediate_nums = np.outer((intermediate_z * ffn_z).reshape(-1), hidden_z).sum().item()

        remaining_model_size = head_nums * self.dim_per_encoder_head * 4 + intermediate_nums * 2
        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}
        # Not multiplied with each other
        results["head_layers"] = head_layer_z.reshape(-1).astype(int).tolist()
        results["ffn_layers"] = ffn_z.reshape(-1).astype(int).tolist()
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size
        
        logger.info(f"remaining_head_layers: {head_layer_z}")
        logger.info(f"remaining_ffn_layers: {ffn_z}")
        logger.info(f"remaining_hidden_dims: {remaining_hidden_dims}")
        logger.info(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        logger.info(f"remaining_head_nums: {remaining_head_nums}")
        logger.info(f"pruned_model_size: {pruned_model_size}")
        logger.info(f"remaining_model_size: {remaining_model_size}")

        return results
        

    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if type != "hidden": # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if type != "hidden_z":
                    zs[type] = torch.stack(zs[type])
        return zs 
    

if __name__ == "__main__":

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("openai/whisper-small")
    l0_module = L0Module(config, lagrangian_warmup=200, target_sparsity=0.5)
 