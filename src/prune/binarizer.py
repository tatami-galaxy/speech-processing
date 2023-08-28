# coding=utf-8
# Copyright 2020-present, AllenAI Authors, University of Illinois Urbana-Champaign,
# Intel Nervana Systems and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Binarizers take a (real value) matrix as input and produce a binary (values in {0,1}) mask of the same shape.
"""

import torch
from torch import autograd
import itertools


def expand_mask(global_mask, in_features, out_features, block_size, sparsity_threshold):

    assert (out_features * in_features) % block_size == 0, "attention matrices need to be divisible by block size squared"
    # total blocks
    #num_blocks = int((out_features * in_features) / (block_size ** 2))
    # row indices where a block starts
    row_ids = [b*block_size for b in range(out_features//block_size)]
    # column indices where a block starts
    col_ids = [b*block_size for b in range(in_features//block_size)]
    # cartesian product
    # each element is a tuple containing the top left position of each block
    block_starts = list(itertools.product(row_ids, col_ids))
    for starts in block_starts:
        # get a block
        block = global_mask[starts[0]:starts[0]+block_size, starts[1]:starts[1]+block_size]
        # count 0's in block
        # block is binary mask
        total = block.numel()
        zeros = total - torch.count_nonzero(block)
        sparsity = zeros/total
        # if block is sparse enough 'prune' entire block
        if sparsity >= sparsity_threshold:
            global_mask[starts[0]:starts[0]+block_size, starts[1]:starts[1]+block_size] = 0  # prune actual mask not 'block'

    return global_mask




class ThresholdBinarizer(autograd.Function):
    """
    Thresholdd binarizer. pruning method : threshold, sigmoied_threshold
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j} > \tau`
    where `\tau` is a real value threshold.

    Implementation is inspired from:
        https://github.com/arunmallya/piggyback
        Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights
        Arun Mallya, Dillon Davis, Svetlana Lazebnik
    """

    @staticmethod
    def forward(
        ctx, 
        inputs: torch.tensor,
        in_features: int,
        out_features: int, 
        threshold: float, 
        sparsity_threshold: float, 
        block_size: int, 
        sigmoid: bool
        ):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The threshold value (in R).
            sigmoid (`bool`)
                If set to ``True``, we apply the sigmoid function to the `inputs` matrix before comparing to `threshold`.
                In this case, `threshold` should be a value between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        nb_elems = inputs.numel()
        nb_min = int(0.005 * nb_elems) + 1  # 0.5%
        if sigmoid:
            mask = (torch.sigmoid(inputs) > threshold).type(inputs.type())

            if sparsity_threshold is not None:
                mask = expand_mask(mask, in_features, out_features, block_size, sparsity_threshold)

        else:
            mask = (inputs > threshold).type(inputs.type())
        if mask.sum() < nb_min:
            # We limit the pruning so that at least 0.5% (half a percent) of the weights are remaining
            # why?
            k_threshold = inputs.flatten().kthvalue(max(nb_elems - nb_min, 1)).values
            mask = (inputs > k_threshold).type(inputs.type())
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None, None, None, None, None, None


class TopKBinarizer(autograd.Function):
    """
    Top-k Binarizer. pruning method : topK
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of S.

    Implementation is inspired from:
        https://github.com/allenai/hidden-networks
        What's hidden in a randomly weighted neural network?
        Vivek Ramanujan*, Mitchell Wortsman*, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
    """

    @staticmethod
    def forward(ctx, inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.flatten().sort(descending=True)
        j = int(threshold * inputs.numel())  # numel -> num of elements

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask

    @staticmethod
    def backward(ctx, gradOutput):
        return gradOutput, None


class MagnitudeBinarizer(object):
    """
    Magnitude Binarizer. pruing method : magnitude
    Computes a binary mask M from a real value matrix S such that `M_{i,j} = 1` if and only if `S_{i,j}`
    is among the k% highest values of |S| (absolute value).

    Implementation is inspired from https://github.com/NervanaSystems/distiller/blob/2291fdcc2ea642a98d4e20629acb5a9e2e04b4e6/distiller/pruning/automated_gradual_pruner.py#L24
    """

    @staticmethod
    def apply(inputs: torch.tensor, threshold: float):
        """
        Args:
            inputs (`torch.FloatTensor`)
                The input matrix from which the binarizer computes the binary mask.
                This input marix is typically the weight matrix.
            threshold (`float`)
                The percentage of weights to keep (the rest is pruned).
                `threshold` is a float between 0 and 1.
        Returns:
            mask (`torch.FloatTensor`)
                Binary matrix of the same size as `inputs` acting as a mask (1 - the associated weight is
                retained, 0 - the associated weight is pruned).
        """
        # Get the subnetwork by sorting the inputs and using the top threshold %
        mask = inputs.clone()
        _, idx = inputs.abs().flatten().sort(descending=True)
        j = int(threshold * inputs.numel())

        # flat_out and mask access the same memory.
        flat_out = mask.flatten()
        flat_out[idx[j:]] = 0
        flat_out[idx[:j]] = 1
        return mask
