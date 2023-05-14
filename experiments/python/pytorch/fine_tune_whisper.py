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
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_scheduler


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'speech-processing':
    root = dirname(root)



common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True)

#print(common_voice)


common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")




common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))



def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"]) #, num_proc=2)



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch



data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


metric = evaluate.load("cer")


model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []


# data loaders
train_dataloader = DataLoader(
    common_voice["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=4,
)
eval_dataloader = DataLoader(
    common_voice["test"],
    collate_fn=data_collator,
    batch_size=4,
)

# optimizer
optimizer = AdamW(
    list(model.parameters()),
    lr=1e-5,
)

num_train_epochs = 5
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# scheduler
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)



#processor.save_pretrained(training_args.output_dir)


device = torch.device("cuda")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

step = 1

for epoch in range(num_train_epochs):

    val_loss = 0
    # Training
    model.train()
    for batch in train_dataloader:

        inputs = batch['input_features'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_features=inputs, labels=labels)
        loss = outputs.loss 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        progress_bar.update(1)


        if step % 500 == 0:
            model.eval()

            for batch in eval_dataloader:
                with torch.no_grad():
                    inputs = batch['input_features'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_features=inputs, labels=labels)
                    val_loss += outputs.loss.item()

                pred_logits = outputs.logits
                pred_ids = np.argmax(pred_logits.detach().cpu().clone().numpy(), axis=-1)
                label_ids = batch['labels']
                label_ids[labels == -100] = processor.tokenizer.pad_token_id

                pred_str = processor.batch_decode(pred_ids)
                # we do not want to group tokens when computing the metrics
                label_str = processor.batch_decode(label_ids, group_tokens=False)

                metric.add_batch(predictions=pred_str, references=label_str)

            cer_result = metric.compute()
            print('step : {}, cer : {}'.format(step, cer_result))
            print(val_loss/len(eval_dataloader))
            print('saving')
            model.save_pretrained(root+'/models/whisper/'+'whisper_small_cv11')

            model.train()
            val_loss = 0

                  
        step += 1
