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
from tqdm.auto import tqdm
from datasets import load_dataset
import transformers, datasets
from transformers import get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq
import torch
import evaluate
from torch.utils.data.dataloader import DataLoader
from transformers import set_seed
import argparse
from accelerate import Accelerator
import nltk
nltk.download('punkt')



def train(args, accelerator):
    # tokenizer, model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    prefix = args.source_prefix if args.source_prefix is not None else ""


    # dataset
    raw_datasets = load_dataset(args.data_dir, args.dataset_config_name)

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

        if args.max_eval_samples is not None:
            raw_datasets["validation"] = raw_datasets["validation"].select(range(args.max_eval_samples))

    # preprocessing the datasets.
    # first we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    
    def preprocess_function(examples):
        inputs = examples[args.text_column]
        targets = examples[args.summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True)

        # pad in collator

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on train dataset",
        )

        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on eval dataset",
        )

        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            num_proc=args.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on test dataset",
        )


    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    

    # Metric
    metric = evaluate.load("rouge")
    

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.train_steps * args.gradient_accumulation_steps
    )


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )


    global_step = 0  # tracks total steps
    total_loss = 0  # total loss before each eval

    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
    }

    accelerator.log({
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gpus": accelerator.state.num_processes
        },
        step=global_step + 1,
    )

    # load from checkpoint
    # check if checkpoint directory passed in
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # if resumed from checkpoint
        # we need to skip steps until we reach the current step
            # ../checkpoint-123 -> int(123)
        steps_completed = int(args.resume_from_checkpoint.split('/')[-1].split('-')[-1])
        global_step = steps_completed
        if args.skip_steps:
            train_dataloader = accelerator.skip_first_batches(train_dataloader, steps_completed) # consider dataset len
            


    # Training

    # main progress bar
    progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process, position=0)
    eval_bar = tqdm(range(len(eval_dataloader)), position=1)

    while True:

        model.train()

        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().item() # for tensorboard 
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)

            if (global_step + 1) % args.eval_steps == 0:
                # eval progress bar
                #eval_bar = tqdm(range(len(eval_dataloader)), position=1)
                model.eval()
                val_loss = 0

                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    # compute metric
                    output_ids = accelerator.unwrap_model(model).generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **gen_kwargs,
                    )

                    # pad_acrss_processes recursively pads the tensors 
                    # in a nested list/tuple/dictionary of tensors from all devices 
                    # to the same size so they can safely be gathered
                    output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
                    label_ids = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                    # gather from all devices
                    output_ids = accelerator.gather(output_ids)  #.cpu().numpy()  # gather_for_metrics
                    label_ids = accelerator.gather(label_ids)  #.cpu().numpy()  # gather_for_metrics

                    # decode ids
                    label_ids[label_ids == -100] = tokenizer.pad_token_id
                    if isinstance(output_ids, tuple):
                        output_ids = output_ids[0]

                    predictions = tokenizer.batch_decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    references = tokenizer.batch_decode(
                        label_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    predictions, references = postprocess_text(predictions, references)

                    metric.add_batch(predictions=predictions, references=references)

                    eval_bar.update(1)

                eval_bar.refresh()
                eval_bar.reset()

                result = metric.compute(use_stemmer=True)
                result = {k: round(v * 100, 4) for k, v in result.items()}

                accelerator.print('step : {}, rouge : {}'.format(global_step + 1, result))
                accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                accelerator.log({
                    "rouge": result,
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    "val_loss": val_loss / len(eval_dataloader)
                },
                step=global_step + 1,
                )

                # save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # saved to folders named `checkpoint-{global_step}`
                # will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # if mixed precision was used, will also save a "scalar.bin" file
                output_dir = f"checkpoint-{global_step + 1}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # save config
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    #model.config.save_pretrained(output_dir)
                    unwrapped_model.config.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

                model.train()
                total_loss = 0

            global_step += 1

            if global_step >= args.train_steps : 
                if args.do_test:
                    ## test here
                    pass
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
        default="google/flan-t5-base",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--data_dir",
        default="cnn_dailymail",
        type=str,
        help="Dataset",
    )
    parser.add_argument(
        "--dataset_config_name",
        default="3.0.0",
        type=str,
        help="config",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        type=str,
        help="checkpoint directory to load model from",
    )
    parser.add_argument(
        "--skip_steps",
        action="store_true",
        help="whether to skip steps in dataloader (checkpoint)"
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        '--max_eval_samples',
        type=int,
        default=None
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--train_steps",
        default=20000,
        type=int,
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--do-test",
        action="store_true",
        help="whether to test or not"
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=os.cpu_count(), # os.cpu_count()
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--lr",
        default=5e-5,
        type=float,
    )
    parser.add_argument(
        "--mixed_precision",
        default='fp16',
        type=str,
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
    )
    parser.add_argument(
        '--max_target_length',
        type=int,
        default=225
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=1
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="article",
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default="highlights",
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text (useful for T5 models).",
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

    # check if output directory is passed in
    if args.output_dir is None:
        model_str = args.model_name_or_path.split('/')[-1]
        data_str = args.data_dir.split('/')[-1]
        args.output_dir = model_str+'-'+data_str   ## dont upload model to repo
    print('output directory set to : {}'.format(args.output_dir))
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    # check if model path is None
    if args.model_name_or_path is None:
        raise ValueError(
            f"pass in model_name_or_path"
        )
    

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.output_dir 
    )
    # to have only one message per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    #if accelerator.is_main_process:
        #datasets.utils.logging.set_verbosity_warning()
        #transformers.utils.logging.set_verbosity_info()
    #else:
        #datasets.utils.logging.set_verbosity_error()
        #transformers.utils.logging.set_verbosity_error()
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.lr,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "per_device_train_batch_size": args.per_device_train_batch_size,
    }

    accelerator.init_trackers('runs', track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    main()

