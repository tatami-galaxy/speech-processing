"""

| Size   | Layers | Width | Heads | Parameters | English-only                                         | Multilingual                                      |
|--------|--------|-------|-------|------------|------------------------------------------------------|---------------------------------------------------|
| tiny   | 4      | 384   | 6     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny.)  |
| base   | 6      | 512   | 8     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)   |
| small  | 12     | 768   | 12    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)  |
| medium | 24     | 1024  | 16    | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium) |
| large  | 32     | 1280  | 20    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)  |

"""

import os
from os.path import dirname, abspath
from tqdm.auto import tqdm
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple
import warnings

import torch
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset, DatasetDict
from datasets import Audio

from whisper_traceable import WhisperForConditionalGeneration
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    set_seed,
)
from transformers.modeling_outputs import BaseModelOutput

#from torch.ao.quantization import get_default_qconfig
from quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping, QConfig
from torch.ao.quantization.utils import is_per_tensor
from torch.ao.quantization.observer import (
    UniformQuantizationObserverBase,
    default_per_channel_weight_observer,
    default_weight_observer,
    default_observer,
)
default_qconfig = QConfig(
    activation=default_observer,
    weight=default_weight_observer
)



# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    


class HistogramObserver(UniformQuantizationObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. ``calculate_qparams`` will calculate scale and zero_point.

    Args:
        bins: Number of bins to use for the histogram
        upsample_rate: Factor by which the histograms are upsampled, this is
                       used to interpolate histograms with varying ranges across observations
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The scale and zero point are computed as follows:

    1. Create the histogram of the incoming inputs.
        The histogram is computed continuously, and the ranges per bin change
        with every new tensor observed.
    2. Search the distribution in the histogram for optimal min/max values.
        The search for the min/max values ensures the minimization of the
        quantization error with respect to the floating point model.
    3. Compute the scale and zero point the same way as in the
        :class:`~torch.ao.quantization.MinMaxObserver`
    """
    histogram: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bins: int = 2048,
        upsample_rate: int = 128,
        dtype: torch.dtype = torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
    ) -> None:
        if not is_per_tensor(qscheme):
            raise NotImplementedError(
                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
                    and torch.per_tensor_affine."
            )
        # bins: The number of bins used for histogram calculation.
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.bins = bins
        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
        self.upsample_rate = upsample_rate

    def _get_norm(
        self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute the norm of the values uniformaly distributed between
        delta_begin and delta_end.
        Currently only L2 norm is supported.

        norm = density * (integral_{begin, end} x^2)
             = density * (end^3 - begin^3) / 3
        """
        norm = (
            delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin
        ) / 3
        return density * norm

    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
        r"""
        Compute the quantization error if we use start_bin to end_bin as the
        min and max to do the quantization.
        """
        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins

        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
        if dst_bin_width == 0.0:
            return 0.0

        src_bin = torch.arange(self.bins, device=self.histogram.device)
        # distances from the beginning of first dst_bin to the beginning and
        # end of src_bin
        src_bin_begin = (src_bin - next_start_bin) * bin_width
        src_bin_end = src_bin_begin + bin_width

        # which dst_bins the beginning and end of src_bin belong to?
        dst_bin_of_begin = torch.clamp(
            torch.div(src_bin_begin, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

        dst_bin_of_end = torch.clamp(
            torch.div(src_bin_end, dst_bin_width, rounding_mode='floor'), 0, self.dst_nbins - 1
        )
        density = self.histogram / bin_width

        norm = torch.zeros(self.bins, device=self.histogram.device)

        delta_begin = src_bin_begin - dst_bin_of_begin_center
        delta_end = dst_bin_width / 2
        norm += self._get_norm(delta_begin,
                               torch.ones(self.bins, device=self.histogram.device) * delta_end,
                               density)

        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
        )

        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2

        delta_begin = -dst_bin_width / 2
        delta_end = src_bin_end - dst_bin_of_end_center
        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)

        return norm.sum().item()

    def _non_linear_param_search(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Non-linear parameter search.

        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
        assert self.histogram.size()[0] == self.bins, "bins mismatch"
        bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = torch.sum(self.histogram).item()
        cSum = torch.cumsum(self.histogram, dim=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")

        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            # decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self._compute_quantization_error(next_start_bin, next_end_bin)

            if norm > norm_min:
                break
            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + bin_width * start_bin
        new_max = self.min_val + bin_width * (end_bin + 1)
        return new_min, new_max

    def _adjust_min_max(
        self, combined_min: torch.Tensor, combined_max: torch.Tensor, upsample_rate: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        # We ensure that:
        # (combined_max - combined_min)/(downsample_rate*Nbins) = (max - min)/(upsample_rate*Nbins)
        # This allows us to have a common grid of resolution s, where we can align
        # the input histogram
        # start_idx maps min_val to the histogram bin index.

        # Compute the width of histogram bins is a straightforward solution, where
        # hist_bin_width = (self.max_val - self.min_val) / (self.bins * upsample_rate)
        # Underflow happens if the numerator is close to the smallest positive subnormal number of FP32
        # Therefore, we avoid such division operation.
        downsample_rate = int(
            torch.ceil(
                (combined_max - combined_min) * upsample_rate / (self.max_val - self.min_val)
            ).item()
        )
        e = downsample_rate * (self.max_val - self.min_val) / upsample_rate - (combined_max - combined_min)
        start_idx = int(
            torch.round((self.min_val - combined_min) * self.bins * upsample_rate / (self.max_val - self.min_val)).item()
        )
        combined_max = combined_max + e
        combined_min = combined_min
        return combined_min, combined_max, downsample_rate, start_idx

    def _combine_histograms(
        self,
        orig_hist: torch.Tensor,
        new_hist: torch.Tensor,
        upsample_rate: int,
        downsample_rate: int,
        start_idx: int,
        Nbins: int,
    ) -> torch.Tensor:
        # First up-sample the histogram with new data by a factor of L
        # This creates an approximate probability density thats piecewise constant
        upsampled_histogram = new_hist.repeat_interleave(upsample_rate)
        # Now insert the upsampled histogram into the output
        # histogram, which is initialized with zeros.
        # The offset at which the histogram is introduced is determined
        # by the start index as the output histogram can cover a wider range
        histogram_with_output_range = torch.zeros(
            (Nbins * downsample_rate), device=orig_hist.device
        )
        histogram_with_output_range[
            start_idx : Nbins * upsample_rate + start_idx
        ] = upsampled_histogram
        # Compute integral histogram, double precision is needed to ensure
        # that there are no overflows
        integral_histogram = torch.cumsum(
            histogram_with_output_range, 0, dtype=torch.double
        )[downsample_rate - 1 :: downsample_rate]
        # Finally perform interpolation
        shifted_integral_histogram = torch.zeros((Nbins), device=orig_hist.device)
        shifted_integral_histogram[1:Nbins] = integral_histogram[0:-1]
        interpolated_histogram = (
            integral_histogram - shifted_integral_histogram
        ) / upsample_rate
        orig_hist = orig_hist + interpolated_histogram.to(torch.float)
        return orig_hist

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:

        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()
        min_val = self.min_val
        max_val = self.max_val
        same_values = min_val.item() == max_val.item()
        is_uninitialized = min_val == float("inf") and max_val == float("-inf")
        if is_uninitialized or same_values:
            min_val, max_val = torch.aminmax(x)
            self.min_val.resize_(min_val.shape)
            self.min_val.copy_(min_val)
            self.max_val.resize_(max_val.shape)
            self.max_val.copy_(max_val)
            assert (
                min_val.numel() == 1 and max_val.numel() == 1
            ), "histogram min/max values must be scalar."
            # casting input to float
            torch.histc(
                x.to(torch.float), self.bins, min=min_val, max=max_val, out=self.histogram  # type: ignore[arg-type]
            )
        else:
            new_min, new_max = torch.aminmax(x)
            combined_min = torch.min(new_min, min_val)
            combined_max = torch.max(new_max, max_val)
            # combine the existing histogram and new histogram into 1 histogram
            # We do this by first upsampling the histogram to a dense grid
            # and then downsampling the histogram efficiently
            (
                combined_min,
                combined_max,
                downsample_rate,
                start_idx,
            ) = self._adjust_min_max(combined_min, combined_max, self.upsample_rate)
            assert (
                combined_min.numel() == 1 and combined_max.numel() == 1
            ), "histogram min/max values must be scalar."
            # casting input to float
            combined_histogram = torch.histc(
                x.to(torch.float), self.bins, min=combined_min, max=combined_max  # type: ignore[arg-type]
            )
            if combined_min == min_val and combined_max == max_val:
                combined_histogram += self.histogram
            else:
                combined_histogram = self._combine_histograms(
                    combined_histogram,
                    self.histogram,
                    self.upsample_rate,
                    downsample_rate,
                    start_idx,
                    self.bins,
                )

            self.histogram.detach_().resize_(combined_histogram.shape)
            self.histogram.copy_(combined_histogram)
            self.min_val.detach_().resize_(combined_min.shape)
            self.min_val.copy_(combined_min)
            self.max_val.detach_().resize_(combined_max.shape)
            self.max_val.copy_(combined_max)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        is_uninitialized = self.min_val == float("inf") and self.max_val == float(
            "-inf"
        )
        if is_uninitialized:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0], device=self.min_val.device.type), torch.tensor([0], device=self.min_val.device.type)
        assert self.bins == len(self.histogram), (
            "The number of bins in histogram should be equal to the number of bins "
            "supplied while making this observer"
        )

        new_min, new_max = self._non_linear_param_search()

        return self._calculate_qparams(new_min, new_max)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + "min_val"] = self.min_val
        destination[prefix + "max_val"] = self.max_val

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 3:
            # if min_val and max_val are not initialized, update their shape
            # to account for the differences between v2 and v3
            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
            if min_val_name in state_dict:
                if state_dict[min_val_name].shape == torch.Size([0]):
                    state_dict[min_val_name] = torch.tensor(float("inf"))
            if max_val_name in state_dict:
                if state_dict[max_val_name].shape == torch.Size([0]):
                    state_dict[max_val_name] = torch.tensor(float("-inf"))

        local_state = ["min_val", "max_val"]
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}"


    

def get_default_qconfig(backend='x86', version=0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: " + str(backend) +
            f" not supported. backend must be one of {supported_backends}"
        )

    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_weight_observer)
        elif backend == 'onednn':
            if not torch.cpu._is_cpu_support_vnni():
                warnings.warn(
                    "Default qconfig of oneDNN backend with reduce_range of false may have accuracy issues "
                    "on CPU without Vector Neural Network Instruction support.")
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_per_channel_weight_observer)
        elif backend == 'x86':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        else:
            # won't reach
            qconfig = default_qconfig
    else:
        raise AssertionError("Version number: " + str(version) +
                             " in get_default_qconfig is not supported. Version number must be 0")

    return qconfig


    

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")
    


def load_model(args):
    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task=args.task)

    # model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    #model.config.forced_decoder_ids = None
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task=args.task)
    model.config.suppress_tokens = []

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False


    model.to("cpu")


    return model, feature_extractor, tokenizer, processor



def load_data(args, model, feature_extractor, tokenizer):


    # dataset
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(args.data_dir, args.data_lang, split="train+validation")
    common_voice["test"] = load_dataset(args.data_dir, args.data_lang, split="test")

    # remove unused columns
    common_voice = common_voice.remove_columns(
        [
            "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
        ]
    )

    # select small dataset for testing
    if args.max_train_samples is not None:
        common_voice["train"] = common_voice["train"].select(range(args.max_train_samples))

    if args.max_test_samples is not None:
        common_voice["test"] = common_voice["test"].select(range(args.max_test_samples))

    # resample to 16kHz
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=args.sampling_rate))


    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(model.config, "model_type", None) == "whisper"
        and getattr(model.config, "apply_spec_augment", False)
        and getattr(model.config, "mask_time_prob", 0) > 0
    )
    # get attention mask for observers
    #forward_attention_mask = True

    # other hyperparameters
    max_input_length = args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = args.min_duration_in_seconds * feature_extractor.sampling_rate
    #audio_column_name = args.audio_column_name
    #num_workers = args.num_workers
    #text_column_name = args.text_column_name
    #do_lower_case = args.do_lower_case
    model_input_name = feature_extractor.model_input_names[0]

    # function to vectorize dataset
    #def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        #audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        #features = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask=True)
        #batch["input_features"] = features.input_features[0]
        #batch["attention_mask"] = features.attention_mask[0]

        # encode target text to label ids 
        #batch["labels"] = tokenizer(batch["sentence"]).input_ids

        #return batch
    
    def prepare_dataset(batch):
        # process audio
        sample = batch["audio"]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_attention_mask=forward_attention_mask
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        # process targets
        input_str = batch["sentence"]  # do lower
        batch["labels"] = tokenizer(input_str).input_ids
        return batch
    
    
    # vectorize dataset
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=args.num_workers)


    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    common_voice = common_voice.filter(
        is_audio_in_length_range,
        num_proc=args.num_workers,
        input_columns=["input_length"],
    )

    return common_voice, forward_attention_mask


# calibrate with generate? or train data?
def calibrate(model, data_loader):

    eval_bar = tqdm(range(len(data_loader)), position=0)
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            model(**batch)
            eval_bar.update(1)
            
    return

    



def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default="openai/whisper-small",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--data_dir",
        default="mozilla-foundation/common_voice_11_0",
        type=str,
        help="Dataset",
    )
    parser.add_argument(
        "--sampling_rate",
        default=16000,
        type=int,
        help="sampling rate",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--model_lang",
        default='hindi',
        type=str,
    )
    parser.add_argument(
        "--task",
        default='transcribe',
        type=str,
    )
    parser.add_argument(
        "--data_lang",
        default='hi',
        type=str,
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None, # os.cpu_count()
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        '--generation_max_length',
        type=int,
        default=225
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=1
    )
    parser.add_argument(
        '--max_duration_in_seconds',
        type=float,
        default=20.0
    )
    parser.add_argument(
        '--min_duration_in_seconds',
        type=float,
        default=0.0
    )


    # parse args
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # check if data path exists
    if args.data_dir is None:
        raise ValueError(
            f"pass in dataset directory"
        )
    ##args.processed_data_dir = root+'/data/processed/'+args.processed_data_dir+'/'
    #if not os.path.isdir(args.data_dir):
        #raise ValueError(
            #f"data directory does not exist"
        #)

    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str
    print('output directory set to : {}'.format(args.output_dir))
    ##if not os.path.isdir(args.output_dir):
        ##os.mkdir(args.output_dir)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )

    # model
    model, feature_extractor, tokenizer, processor = load_model(args)

    # data
    common_voice, forward_attention_mask = load_data(args, model, feature_extractor, tokenizer)

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )

    # data loaders
    train_dataloader = DataLoader(
        common_voice["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
    )
    test_dataloader = DataLoader(
        common_voice["test"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )


    # config for static quantization
    qconfig = get_default_qconfig("x86")
    #print(qconfig)
    #quit()
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # example inputs to set up observers
    example_inputs = (next(iter(train_dataloader)))
    #print(example_inputs)

    # concrete args for tracing control flow
    concrete_args={
        'output_hidden_states': False,
        'output_attentions': False,
        'return_dict': True,
        'use_cache': True, 
        'decoder_inputs_embeds': None,
        'head_mask': None,
        'past_key_values': None, # cant be None for fast generation
        'head_mask': None,
        'decoder_head_mask': None,
        'cross_attn_head_mask': None,
    }

    # quantize model
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs, concrete_args=concrete_args)

    update_dict = {}
    for key, val in concrete_args.items():
        new_key = key+'_1'
        update_dict[new_key] = val
    
    # update nodes
    for node in prepared_model.graph.nodes:
        if node.name in update_dict:
            node.update_arg(0, update_dict[node.name])

    prepared_model.recompile()

    for node in prepared_model.graph.nodes:
        print(node.format_node())
    quit()


    calibrate(prepared_model, test_dataloader)
    quantized_model = convert_fx(prepared_model)

    

    print("Size of model before quantization")
    print_size_of_model(model)
    print("Size of model after quantization")
    print_size_of_model(quantized_model)





            


if __name__ == "__main__":

    main()

