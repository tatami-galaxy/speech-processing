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
            "hidden",  # common for encoder and decoder
            "en_head",
            "en_mha",
            "en_ffn_dim",
            "en_ffn",
            "de_self_head",
            "de_self_mha",
            "de_cross_head",
            "de_cross_mha",
            "de_ffn_dim",
            "de_ffn",
        ]
        
        # model parameters
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model  # hidden_size

        # layers
        self.num_hidden_layers_en = config.encoder_layers
        self.num_hidden_layers_de  = config.decoder_layers
        # heads
        # num_attention_heads / layer
        self.encoder_attention_heads = config.encoder_attention_heads  
        # 12 self attn heads, 12 cross attn heads
        self.decoder_attention_heads = config.decoder_attention_heads
        # ffn dims
        self.encoder_ffn_dim = config.encoder_ffn_dim  # intermediate_size
        self.decoder_ffn_dim = config.decoder_ffn_dim  # intermediate_size
        # number of ffns
        self.ffn_num_per_layer = 1  # linear -> activation -> linear
        # head dims
        # used in calculate_model_size ()
        self.dim_per_encoder_head = self.d_model // self.encoder_attention_heads
        self.dim_per_decoder_head = self.d_model // self.decoder_attention_heads

        # same number of heads, head size for encoder, decoder
        # 4 attention weight matrices. 4 weights, 3 biases
        # no bias for k_proj
        # self attn and cross attn for decoder added separately
        self.params_per_head_layer = self.d_model * self.d_model * 4  + self.d_model * 3
        # same number of heads for en self_attn, de self_attn, de cross_attn
        self.params_per_head =  self.params_per_head_layer // self.encoder_attention_heads 
        
        # same intermediate size for encoder, decoder (change if needed)
        # weights, biases for each ffn layer
        # same for encoder and decoder
        # 2 weight matrices + 2 biases
        self.params_per_ffn_layer = self.d_model * config.encoder_ffn_dim * 2  + self.d_model + config.encoder_ffn_dim 
        self.params_per_ffn_dim = self.params_per_ffn_layer // self.encoder_ffn_dim

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.prunable_model_size = 0 

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        # Q parameters
        self.z_logas = {}
        # multiply with E to get expected model size
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


    # called after init
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
            # what does this mean?
            #parameter_per_dim=self.d_model * 4 + self.d_model* 4 * 2,
            parameter_per_dim=None,
            # how is size and shape used?
            size=self.d_model, shape=[self.d_model]
        )
        self.reset_loga(self.hidden_loga, mean=10)  # different mean?


    def initialize_head(self):
        # loga
        self.en_head_loga = self.initialize_parameters(self.encoder_attention_heads, self.num_hidden_layers_en)
        self.de_self_head_loga = self.initialize_parameters(self.decoder_attention_heads, self.num_hidden_layers_de)
        self.de_cross_head_loga = self.initialize_parameters(self.decoder_attention_heads, self.num_hidden_layers_de)
        self.reset_loga(self.en_head_loga, mean=10)
        self.reset_loga(self.de_self_head_loga, mean=10)
        self.reset_loga(self.de_cross_head_loga, mean=10)
        # encoder
        self.add_one_module(self.en_head_loga, type="en_head", 
                            parameter_per_dim=self.params_per_head,
                            # encoder self attn heads
                            size=self.encoder_attention_heads,
                            # how is size and shape used?
                            shape=[self.num_hidden_layers_en, 1, self.encoder_attention_heads, 1, 1])
        # decoder (self and cross)
        self.add_one_module(self.de_self_head_loga, type="de_self_head", 
                            parameter_per_dim=self.params_per_head,
                            # decoder self attn heads
                            size=self.decoder_attention_heads,
                            # how is size and shape used?
                            shape=[self.num_hidden_layers_de, 1, self.decoder_attention_heads, 1, 1])
        self.add_one_module(self.de_cross_head_loga, type="de_cross_head", 
                            parameter_per_dim=self.params_per_head,
                            # decoder cross attn heads
                            size=self.decoder_attention_heads,
                            # how is size and shape used?
                            shape=[self.num_hidden_layers_de, 1, self.decoder_attention_heads, 1, 1])
        
        # 2 sets of heads(self, cross) for decoder
        self.prunable_model_size += self.params_per_head * (self.num_hidden_layers_en + self.num_hidden_layers_de) * (self.encoder_attention_heads + 2*self.decoder_attention_heads)


    def initialize_mha(self):
        n_layer_en = self.num_hidden_layers_en
        n_layer_de = self.num_hidden_layers_de
        self.en_headlayer_loga = self.initialize_parameters(n_layer_en)
        self.de_self_headlayer_loga = self.initialize_parameters(n_layer_de)
        self.de_cross_headlayer_loga = self.initialize_parameters(n_layer_de)
        self.reset_loga(self.en_headlayer_loga, mean=10)
        self.reset_loga(self.de_self_headlayer_loga, mean=10)
        self.reset_loga(self.de_cross_headlayer_loga, mean=10)
        # encoder
        self.add_one_module(self.en_headlayer_loga, type="en_mha", 
                            parameter_per_dim=self.params_per_head * self.encoder_attention_heads, size=1,
                            shape=[n_layer_en])
        # decoder (self and cross)
        self.add_one_module(self.de_self_headlayer_loga, type="de_self_mha", 
                            parameter_per_dim=self.params_per_head * self.decoder_attention_heads, size=1,
                            shape=[n_layer_de])
        self.add_one_module(self.de_cross_headlayer_loga, type="de_cross_mha", 
                            parameter_per_dim=self.params_per_head * self.decoder_attention_heads, size=1,
                            shape=[n_layer_de])
        # no change in prunable_model_size since all heads already considered
        #logger.info(f"Initialized layerwise mha. Prunable_model_size = {self.prunable_model_size}")


    def initialize_ffn_dim(self):
        self.en_int_loga = self.initialize_parameters(self.encoder_ffn_dim, self.num_hidden_layers_en)
        self.de_int_loga = self.initialize_parameters(self.decoder_ffn_dim, self.num_hidden_layers_de)
        # encoder
        self.add_one_module(self.en_int_loga, type="en_ffn_dim", 
                            parameter_per_dim=self.params_per_ffn_dim, size=self.encoder_ffn_dim,
                            shape=[self.num_hidden_layers_en, 1, 1, self.encoder_ffn_dim])
        # decoder
        self.add_one_module(self.de_int_loga, type="de_ffn_dim", 
                            parameter_per_dim=self.params_per_ffn_dim, size=self.decoder_ffn_dim,
                            shape=[self.num_hidden_layers_de, 1, 1, self.decoder_ffn_dim])
    
        self.reset_loga(self.en_int_loga)
        self.reset_loga(self.de_int_loga)

        self.prunable_model_size += self.params_per_ffn_layer * (self.num_hidden_layers_en + self.num_hidden_layers_de)
        #logger.info(f"Initialized ffn dim. Prunable_model_size = {self.prunable_model_size}")


    def initialize_ffn(self):
        n_layer_en = self.num_hidden_layers_en
        n_layer_de = self.num_hidden_layers_de
        self.en_intlayer_loga = self.initialize_parameters(n_layer_en)
        self.de_intlayer_loga = self.initialize_parameters(n_layer_de)
        # encoder
        self.add_one_module(self.en_intlayer_loga, type="en_ffn", 
                            parameter_per_dim=None, size=self.ffn_num_per_layer,  # 1
                            shape=[n_layer_en])
        # decoder
        self.add_one_module(self.de_intlayer_loga, type="de_ffn", 
                            parameter_per_dim=None, size=self.ffn_num_per_layer,  # 1
                            shape=[n_layer_de])
        self.reset_loga(self.en_intlayer_loga, mean=10)
        self.reset_loga(self.de_intlayer_loga, mean=10)
        #logger.info(f"Initialized ffn. Prunable_model_size = {self.prunable_model_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    
    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])


    # CDF of the stretched concrete distribution
    def cdf_qz(self, x, loga):
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)


    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a


    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size


    def transform_scores_for_head(self):
        # 1 - Q(s<=0)
        en_mha_score = 1 - self.cdf_qz(0, self.en_headlayer_loga)
        de_self_mha_score = 1 - self.cdf_qz(0, self.de_self_headlayer_loga)
        de_cross_mha_score = 1 - self.cdf_qz(0, self.de_cross_headlayer_loga)
        en_head_score = 1 - self.cdf_qz(0, self.en_head_loga) # 12 * 12
        de_self_head_score = 1 - self.cdf_qz(0, self.de_self_head_loga)
        de_cross_head_score = 1 - self.cdf_qz(0, self.de_cross_head_loga)
       
        if en_mha_score is not None:
            en_mha_score = en_mha_score.view(-1, 1, 1)  # 12 * 1 * 1
        if de_self_mha_score is not None:
            de_self_mha_score = de_self_mha_score.view(-1, 1, 1)  # 12 * 1 * 1
        if de_cross_mha_score is not None:
            de_cross_mha_score = de_cross_mha_score.view(-1, 1, 1)  # 12 * 1 * 1

        en_head_score = en_head_score.unsqueeze(-1)   # 12 * 12 * 1
        de_self_head_score = de_self_head_score.unsqueeze(-1)   # 12 * 12 * 1
        de_cross_head_score = de_cross_head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return en_mha_score, de_self_mha_score, de_cross_mha_score, en_head_score, de_self_head_score, de_cross_head_score


    # change #
    def get_num_parameters_for_ffn(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters


    # change #
    def get_num_parameters_and_constraint(self): #! calculate the current sparsity
        num_parameters = 0
        
        # attn scores
        # 12 * 1 * 1
        # 12 * 12 * 1
        en_mha_score, de_self_mha_score, de_cross_mha_score, en_head_score, de_self_head_score, de_cross_head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if any(s is not None for s in (en_mha_score, de_self_mha_score, de_cross_mha_score)):
            # head score x mha score
            if en_mha_score is not None:
                en_head_score = (en_mha_score * en_head_score).reshape(-1)
            if de_self_mha_score is not None:
                de_self_head_score = (de_self_mha_score * de_self_head_score).reshape(-1)
            if de_cross_mha_score is not None:
                de_cross_head_score = (de_cross_mha_score * de_cross_head_score).reshape(-1)
        else:
            en_head_score = en_head_score.reshape(-1)
            de_self_head_score = de_self_head_score.reshape(-1)
            de_cross_head_score = de_cross_head_score.reshape(-1)

        # E (score) x num_parameters
        num_parameters += torch.sum(torch.outer(hidden_score, en_head_score)) * self.parameters_per_dim["en_head"] / self.d_model
        num_parameters += torch.sum(torch.outer(hidden_score, de_self_head_score)) * self.parameters_per_dim["de_self_head"] / self.d_model
        num_parameters += torch.sum(torch.outer(hidden_score, de_cross_head_score)) * self.parameters_per_dim["de_cross_head"] / self.d_model

        # ffn scores
        en_intlayer_score = 1 - self.cdf_qz(0, self.en_intlayer_loga)  # 12
        de_intlayer_score = 1 - self.cdf_qz(0, self.de_intlayer_loga)  # 12
        en_int_score = 1 - self.cdf_qz(0, self.en_int_loga)  # 12 * 3072
        de_int_score = 1 - self.cdf_qz(0, self.de_int_loga)  # 12 * 3072

        en_intlayer_score = en_intlayer_score.unsqueeze(-1)
        de_intlayer_score = de_intlayer_score.unsqueeze(-1)

        en_int_score = (en_intlayer_score * en_int_score).reshape(-1)
        de_int_score = (de_intlayer_score * de_int_score).reshape(-1)

        # why times 2?
        #num_parameters += torch.sum(torch.outer(hidden_score, en_int_score)) * 2
        #num_parameters += torch.sum(torch.outer(hidden_score, de_int_score)) * 2

        # E (score) x num_parameters
        num_parameters += torch.sum(torch.outer(hidden_score, en_int_score)) * self.parameters_per_dim["en_ffn_dim"] / self.d_model
        num_parameters += torch.sum(torch.outer(hidden_score, de_int_score)) * self.parameters_per_dim["de_ffn_dim"] / self.d_model
        
        return num_parameters



    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity


    def lagrangian_regularization(self, pruned_steps):

        target_sparsity = self.target_sparsity
        expected_size = self.get_num_parameters_and_constraint() #! calculate \bar s
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
        for _type in self.types:
            name = _type  #[:-2]
            z = zs.get(_type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                z = z.squeeze().detach().cpu().numpy() > 0  # new_z
            numpified_zs[name] = z  # new_z
        return numpified_zs


    def calculate_model_size_np(self, zs):

        numpified_zs = self.get_z_from_zs(zs)

        hidden_z = numpified_zs["hidden"]
        en_head_z = numpified_zs["en_head"]
        en_mha_z = numpified_zs["en_mha"].reshape(-1, 1)
        en_ffn_dim_z = numpified_zs["en_ffn_dim"]
        en_ffn_z = numpified_zs["en_ffn"].reshape(-1, 1)
        de_self_head_z = numpified_zs["de_self_head"]
        de_self_mha_z = numpified_zs["de_self_mha"].reshape(-1, 1)
        de_cross_head_z = numpified_zs["de_cross_head"]
        de_cross_mha_z = numpified_zs["de_cross_mha"].reshape(-1, 1)
        de_ffn_dim_z = numpified_zs["de_ffn_dim"]
        de_ffn_z = numpified_zs["de_ffn"].reshape(-1, 1)

        # hidden dims
        remaining_hidden_dims = hidden_z.sum().item()
        # ffns
        remaining_en_ffn_nums = en_ffn_dim_z.reshape(self.num_hidden_layers_en, self.encoder_ffn_dim).sum(-1).tolist()
        remaining_de_ffn_nums = de_ffn_dim_z.reshape(self.num_hidden_layers_de, self.decoder_ffn_dim).sum(-1).tolist()
        # heads
        remaining_en_head_nums = en_head_z.reshape(
            self.num_hidden_layers_en, self.encoder_attention_heads).sum(-1).tolist()
        remaining_de_self_head_nums = de_self_head_z.reshape(
            self.num_hidden_layers_de, self.decoder_attention_heads).sum(-1).tolist()
        remaining_de_cross_head_nums = de_cross_head_z.reshape(
            self.num_hidden_layers_de, self.decoder_attention_heads).sum(-1).tolist()

        #en_head_nums = np.outer((en_head_z * en_mha_z).reshape(-1), hidden_z).sum().item()
        #de_self_head_nums = np.outer((de_self_head_z * de_self_mha_z).reshape(-1), hidden_z).sum().item()
        #de_cross_head_nums = np.outer((de_cross_head_z * de_cross_mha_z).reshape(-1), hidden_z).sum().item()
        #en_ffn_nums = np.outer((en_ffn_dim_z * en_ffn_z).reshape(-1), hidden_z).sum().item()
        #de_ffn_nums = np.outer((de_ffn_dim_z * de_ffn_z).reshape(-1), hidden_z).sum().item()

        en_head_nums = (en_head_z * en_mha_z).reshape(-1).sum().item()
        de_self_head_nums = (de_self_head_z * de_self_mha_z).reshape(-1).sum().item()
        de_cross_head_nums = (de_cross_head_z * de_cross_mha_z).reshape(-1).sum().item()
        en_ffn_nums = (en_ffn_dim_z * en_ffn_z).reshape(-1).sum().item()
        de_ffn_nums = (de_ffn_dim_z * de_ffn_z).reshape(-1).sum().item()

        #remaining_model_size = en_head_nums * self.dim_per_encoder_head * 4 + en_ffn_nums * 2
        remaining_model_size = en_head_nums * self.dim_per_encoder_head + en_ffn_nums 
        #remaining_model_size += (de_self_head_nums + de_cross_head_nums) * self.dim_per_decoder_head * 4 + de_ffn_nums  * 2
        remaining_model_size += (de_self_head_nums + de_cross_head_nums) * self.dim_per_decoder_head + de_ffn_nums

        ## probably incorrect ##
        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}
        # Not multiplied with each other
        results["en_mha"] = en_mha_z.reshape(-1).astype(int).tolist()
        results["de_self_mha"] = de_self_mha_z.reshape(-1).astype(int).tolist()
        results["de_cross_mha"] = de_cross_mha_z.reshape(-1).astype(int).tolist()

        results["en_ffn_layers"] = en_ffn_z.reshape(-1).astype(int).tolist()
        results["de_ffn_layers"] = de_ffn_z.reshape(-1).astype(int).tolist()

        results["hidden_dims"] = remaining_hidden_dims

        results["en_ffn_dims"] = remaining_en_ffn_nums
        results["de_ffn_dims"] = remaining_de_ffn_nums

        results["en_head_nums"] = remaining_en_head_nums
        results["de_self_head_nums"] = remaining_de_self_head_nums
        results["de_cross_head_nums"] = remaining_de_cross_head_nums

        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size

        return results
    

    def calculate_model_size(self, zs):

        hidden_z = zs["hidden_z"]
        en_head_z = zs["en_head_z"]
        en_mha_z = zs["en_mha_z"].reshape(-1, 1)
        en_ffn_dim_z = zs["en_ffn_dim_z"]
        en_ffn_z = zs["en_ffn_z"].reshape(-1, 1)
        de_self_head_z = zs["de_self_head_z"]
        de_self_mha_z = zs["de_self_mha_z"].reshape(-1, 1)
        de_cross_head_z = zs["de_cross_head_z"]
        de_cross_mha_z = zs["de_cross_mha_z"].reshape(-1, 1)
        de_ffn_dim_z = zs["de_ffn_dim_z"]
        de_ffn_z = zs["de_ffn_z"].reshape(-1, 1)

        # hidden dims
        remaining_hidden_dims = torch.count_nonzero(hidden_z).item()
        # ffn dims
        remaining_en_ffn_dims = torch.count_nonzero(
            en_ffn_dim_z.reshape(self.num_hidden_layers_en, self.encoder_ffn_dim), dim=1)
        remaining_de_ffn_dims = torch.count_nonzero(
            de_ffn_dim_z.reshape(self.num_hidden_layers_de, self.decoder_ffn_dim), dim=1)
        # ffns
        remaining_en_ffns = torch.count_nonzero(en_ffn_z, dim=1)
        remaining_de_ffns = torch.count_nonzero(de_ffn_z, dim=1)
        # heads
        remaining_en_head_nums = torch.count_nonzero(
            en_head_z.reshape(self.num_hidden_layers_en, self.encoder_attention_heads), dim=1)
        remaining_de_self_head_nums = torch.count_nonzero(
            de_self_head_z.reshape(self.num_hidden_layers_de, self.decoder_attention_heads), dim=1)
        remaining_de_cross_head_nums = torch.count_nonzero(
            de_cross_head_z.reshape(self.num_hidden_layers_de, self.decoder_attention_heads), dim=1)
        # mha
        remaining_en_mha_nums = torch.count_nonzero(en_mha_z, dim=1)
        remaining_de_self_mha_nums = torch.count_nonzero(de_self_mha_z, dim=1)
        remaining_de_cross_mha_nums = torch.count_nonzero(de_cross_mha_z, dim=1)

        # head params
        total_en_heads = torch.sum(remaining_en_head_nums)
        total_de_heads = torch.sum(remaining_de_self_head_nums) + torch.sum(remaining_de_cross_head_nums)
        remaining_head_params = (total_en_heads + total_de_heads) * self.params_per_head
        # ffn params
        total_en_ffns = torch.sum(remaining_en_ffns)
        total_de_ffns = torch.sum(remaining_de_ffns)
        remaining_ffn_params = (total_en_ffns + total_de_ffns) * self.params_per_ffn_layer
        # remaining model size
        remaining_model_size = remaining_head_params.item() + remaining_ffn_params.item()

        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}

        results["remianing_hidden_dims"] = remaining_hidden_dims

        results["remaining_en_heads"] = remaining_en_head_nums.tolist()
        results["remaining_de_self_heads"] = remaining_de_self_head_nums.tolist()
        results["remaining_de_cross_heads"] = remaining_de_cross_head_nums.tolist()

        results["remaining_en_mha_nums"] = remaining_en_mha_nums.tolist()
        results["remaining_de_self_mha_nums"] = remaining_de_self_mha_nums.tolist()
        results["remaining_de_cross_mha_nums"] = remaining_de_cross_mha_nums.tolist()

        results["remaining_en_ffn_dims"] = remaining_en_ffn_dims.tolist()
        results["remaining_de_ffn_dims"] = remaining_de_ffn_dims.tolist()

        results["remaining_en_ffn_layers"] = remaining_en_ffns.tolist()
        results["remaining_de_ffn_layers"] = remaining_de_ffns.tolist()

        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size

        return results
    

    def calculate_sparsity_distribution(self, zs):
        print(zs["en_head_z"].shape)
        quit()


    # called after init and set_lagrangian_warmup_steps
    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                # parameter of q
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                # hidden is not a per layer sample
                if type != "hidden":
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
 