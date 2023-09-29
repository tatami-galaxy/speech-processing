# Copyright 2020-present, the HuggingFace Inc. team.
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
Count remaining (non-zero) weights in the encoder (i.e. the transformer layers).
Sparsity and remaining weights levels are equivalent: sparsity % = 100 - remaining weights %.
"""
import argparse
from whisper_traceable_masked import MaskedWhisperForConditionalGeneration


def main(args):
    model_name_or_path = args.pruned_model_name_or_path

    #st = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu")
    model = MaskedWhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    remaining_count = 0  # Number of remaining (not pruned) params in the model
    model_count = 0  # Number of params in the model

    print("name".ljust(60, " "), "Remaining Weights %", "Remaining Weight")
    for name, param in model.named_parameters():

        if "embed" in name or "bias" in name or "layer_norm" in name or "conv" in name or "proj_out" in name:
            remaining_count += param.numel()
            model_count += param.numel()
            

        elif "mask_scores" not in name:
            _ones = (param != 0.0).sum().item()
            remaining_count += _ones
            print(name.ljust(60, " "), str(round(100 * _ones / param.numel(), 3)).ljust(20, " "), str(_ones))
            model_count += param.numel()


    print("")
    print("Remaining Weights (global) %: ", 100 * remaining_count / model_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pruned_model_name_or_path",
        type=str,
        required=True,
        help="Folder containing the model that was previously fine-pruned",
    )

    args = parser.parse_args()

    main(args)
