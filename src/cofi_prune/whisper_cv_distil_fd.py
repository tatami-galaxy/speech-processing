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
import random
from os.path import dirname, abspath
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict, load_from_disk
import transformers, datasets
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import get_linear_schedule_with_warmup
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import WhisperForConditionalGeneration
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_scheduler, set_seed
import argparse
from accelerate import Accelerator, DistributedType
from scipy.spatial import distance


# get root directory
root = abspath(__file__)
while root.split('/')[-1] != 'pytorch':
    root = dirname(root)



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
        
class addOnLayers(nn.Module):
      def __init__(self, MAX_TIMESTEP):
          super().__init__()
          self.Layer1 = nn.Linear(2*MAX_TIMESTEP, 64)
          self.Layer2 = nn.Linear(64, 1)
          
      def forward(self, x):
          x = nn.Tanh()(self.Layer1(x))
          x = nn.Sigmoid()(self.Layer2(x))
          return x
      

        

def pdist(a): #(54, 51000)
    epsilon = 1e-7
    aTa = torch.mm(a, a.T) #(54, 54)
    aTa_diag = torch.diag(aTa) #(54)
    mag_prod = torch.clamp(torch.sqrt(aTa_diag) * torch.sqrt(aTa_diag.unsqueeze(-1)), min = epsilon) #(54, 54)
    dot = torch.div(aTa, mag_prod) #(54, 54)
    i = torch.triu_indices(dot.shape[0], dot.shape[1], offset = 1, device = 'cuda') # (1431)
    return dot.flatten().index_select(0, i[0] * dot.shape[0] + i[1])
    
    
 
def distnaceRKD(student_log, teacher_log, args):
    index = 0
    batch_loss = torch.empty(args.train_batch_size, requires_grad=True).to("cuda")
    
    for t_sample, s_sample in zip(teacher_log, student_log):
        tot_loss = 0
        
        s_potential, t_potential = pdist(s_sample), pdist(t_sample)   
        #mu_student, mu_teacher = sum(t_potential)/len(t_potential), sum(s_potential)/len(s_potential)
        #t_potential, s_potential = t_potential/ mu_teacher, s_potential/mu_student
	
        for elem_s, elem_t in zip(s_potential, t_potential):
            if abs(elem_s-elem_t) <= 1:
                loss = 0.5 * abs(elem_s-elem_t)**2
            else:
                loss = abs(elem_s - elem_t) - 0.5
            tot_loss += loss     
           
	        
        batch_loss[index] = tot_loss
        index += 1
    
    return torch.mean(batch_loss)
    

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device='cuda'),
                  torch.zeros(xx.shape).to(device='cuda'),
                  torch.zeros(xx.shape).to(device='cuda'))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)
    
def MMD_loss(student_inp, teacher_inp, kernel, args):
    index = 0
    batch_loss = torch.empty(args.train_batch_size, requires_grad=True).to("cuda")
    for s_sample, t_sample in zip(student_inp, teacher_inp):
        loss = MMD(s_sample, t_sample, kernel)
        batch_loss[index] = loss
        index+=1
    return torch.mean(batch_loss)
    
def instanceLoss(input_logs, target_logs, op = 'cs'):
    '''
    output : model output logits in form of soft predictions
    target : model output in form of index ids 
    ''' 
    losses = torch.empty(input_logs.shape[0], requires_grad=True).to("cuda")         
    
    for i in range(input_logs.shape[0]):
        if op=='cs':
            loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')(
                input=nn.functional.softmax(input_logs[i], dim=-1),
                target=target_logs[i],
            )
        if op=='kl':
            temperature=2
            loss = torch.nn.functional.kl_div(
                input=nn.functional.log_softmax(input_logs[i]/temperature, dim=-1),
                target=nn.functional.softmax(target_logs[i]/temperature, dim=-1),
                reduction='batchmean',
            )*temperature**2
        losses[i] = loss
           
    return losses

def shuffle_groups(dataset, group_size=1000):
    size = dataset['train'].num_rows
    for i in range(0, size, group_size):
        start_index, end_index = i, min(size, i+group_size)
        portion_to_shuffle = dataset['train'][start_index:end_index]
        random.shuffle(portion_to_shuffle)
        dataset['train'][start_index:end_index] = portion_to_shuffle
    return dataset

def getDecoderLayers(model):
    new_layers = torch.nn.ModuleList()

    new_layers.append(model.model.decoder.layers[0])
    new_layers.append(model.model.decoder.layers[-1])

    return new_layers
	

def train(args, accelerator):
    if args.student_name_or_path is None: args.student_name_or_path = args.model_name_or_path
    # extractor, tokenizer, processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language=args.model_lang, task="transcribe")
    # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
    tokenizer.set_prefix_tokens(language=args.model_lang, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.model_lang, task="transcribe")

    # model
    model = WhisperForConditionalGeneration.from_pretrained(args.student_name_or_path, low_cpu_mem_usage=True)
    model.config.forced_decoder_ids = None
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task="transcribe")
    model.config.suppress_tokens = []
    model.config.output_hidden_states = True
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    if args.distill:
        # teacher
        teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher_name_or_path)
        teacher.config.forced_decoder_ids = None
        teacher.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.model_lang, task="transcribe")
        teacher.config.suppress_tokens = []
        teacher.config.output_hidden_states = True
        #model.model.decoder.layers = getDecoderLayers(teacher) ## ""Shrink and Fine Tune"" (Distil-Whisper Paper) (only used if we use a smaller variant of a model{by removing layer})
        

        if teacher.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        print("Teacher model accessed and created")

    # dataset/
    common_voice = DatasetDict()
    common_voice["train"] = load_from_disk(args.train_data_dir)
    common_voice["test"] = load_from_disk(args.test_data_dir)

    with accelerator.main_process_first():
        # remove unused columns
        common_voice = common_voice.remove_columns(
            [
                "accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"
            ]
        )

        # resample to 16kHz
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=args.sampling_rate))


    # function to vectorize dataset
    def prepare_dataset(batch):
        audio = batch["audio"]

        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch
    
    with accelerator.main_process_first():
        common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"]) #, num_proc=2)



    # cer metric
    metric = evaluate.load("./evaluate/metrics/cer/cer.py")

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # data loaders
    train_dataloader = DataLoader(
        common_voice["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
        worker_init_fn=np.random.seed(404)
    )
    eval_dataloader = DataLoader(
        common_voice["test"],
        collate_fn=data_collator,
        batch_size=args.eval_batch_size,
    )

    #optimizer

    optimizer = AdamW(
        list(model.parameters()),
        lr=args.lr,
    )


    # optimizer
    '''
    learnable alpha
    if args.distill:
       MAX_TIMESTEP=0
       for batch in train_dataloader:
           if batch['labels'].shape[1] > MAX_TIMESTEP:
               MAX_TIMESTEP = batch['labels'].shape[1]
       
    addOn = addOnLayers(MAX_TIMESTEP).to('cuda')
    optimizer = AdamW(
        list(model.parameters())+list(addOn.parameters()),
        lr=args.lr,
    )
    '''

    # scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.train_steps
    )
    
    #alpha = 0.8
    alpha_increament = 0
	

    if args.distill:
	    model, teacher, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                   model, teacher, optimizer, train_dataloader, eval_dataloader, lr_scheduler
            )
    else:
    	model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
                   model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

    ##
    global_step = 0
    total_loss = 0
    total_s_loss = 0
    total_d_loss = 0
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        if args.skip_steps:
            # ../checkpoint-123 -> int(123)
            steps_completed = int(args.resume_from_checkpoint.split('/')[-1].split('-')[-1])
            train_dataloader = accelerator.skip_first_batches(train_dataloader, steps_completed) # consider dataset len
            global_step = steps_completed


    # Training

    # main progress bar
    progress_bar = tqdm(range(global_step, args.train_steps), disable=not accelerator.is_main_process)
    
    min_cer = 0
    if args.distill: 
       alpha = args.alpha_distil
       if args.set_cl_params:
          t_ls = []
          
          for batch in tqdm(train_dataloader, total=len(train_dataloader)):
              t_outputs = teacher(**batch)
              t_logits = t_outputs.logits
              t_ls.extend(instanceLoss(t_logits, batch['labels'], 'cs'))
              
          args.tau, args.max_tau  = min(t_ls), max(t_ls)
              

    print("***************************CONFIGURATIONS*************************")
    print("Base Model Path : {}".format(args.model_name_or_path))
    print("Student Model Path : {}".format(args.student_name_or_path))
    print("Teacher Model Path : {}".format(args.teacher_name_or_path))
    print("Temperature : {}".format(args.temperature))
    print("Starting alpha_distil : {}".format(args.alpha_distil))
    print("Starting alpha_self : {}".format(args.alpha_ce))
    print("Dataset  : {}".format(args.train_data_dir))
    print("******************************************************************")
    
    while True:

        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                if args.distill:
                    with torch.no_grad():
                        t_outputs = teacher(**batch)
                        t_logits = t_outputs.logits
                        t_preds = torch.argmax(nn.functional.softmax(t_logits, dim=-1), -1)
                    
            
                # student
                s_outputs = model(**batch)
                ########################################################
                # CER always 1 by the aproach, there might be some error in decoding
                #s_outputs = model(**batch, decoder_input_ids=t_preds) ## "Pseudo Labelling" ~Distill-Whisper
                ########################################################
                s_logits = s_outputs.logits
                s_loss = s_outputs.loss
                
                #######CE Loss between Teacher and Student##########
                #s_preds_t = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')(  
                #                input=s_logits,
                #                target=t_preds,
                #                )
                ####################################################
                
                
                if args.distill:
                    
                    #print("Student Scores : {} \n Teacher Scores : {} \n Labels : {} \n".format(torch.nn.functional.softmax(t_logits), torch.nn.functional.softmax(t_logits), batch['labels']))    
                    #torch.save(torch.nn.functional.softmax(t_logits, dim=-1), "./logs/t_logits.pt")
                    #torch.save(torch.nn.functional.softmax(s_logits, dim=-1), "./logs/s_logits.pt")
                    #torch.save(batch['labels'], "./logs/labels.pt")
                    ############kl loss of logits##################    
                    #d_loss = nn.functional.kl_div(
                    #    input=s_logits / args.temperature,
                    #    target=t_logits / args.temperature,
                    #    reduction="batchmean",
                    #) * (args.temperature**2)
                    ###############################################
                    
                    #############kl loss of logits softmax################
                    
                    SCALING_FACTOR=1
                    d_loss = nn.functional.kl_div(
                    	input=nn.functional.log_softmax(s_logits/args.temperature, dim=-1),
                    	target=nn.functional.softmax(t_logits/args.temperature, dim=-1),  #TEMPERATURE REMOVED
                     	reduction="batchmean",
                     )*(args.temperature**2)
                    d_loss = d_loss/s_logits.shape[1]
                    d_loss = d_loss/SCALING_FACTOR
                    
                    #######################################################

                    ###########################FEATURE LOSS#################################
                    #f_dec_loss, f_enc_loss = 0, 0
                    #for i in range(len(model.model.encoder.layers)):
                    #    print(s_outputs.decoder_hidden_states[i].shape, t_outputs.decoder_hidden_states[i*3].shape)
                    #    print(s_outputs.encoder_hidden_states[i].shape, t_outputs.encoder_hidden_states[i*3].shape)
                    #    #f_dec_loss += mse_loss(outputs.decoder_hidden_layers[i], t_outputs.decoder_hidden_layers[i*3])
                    #    #f_enc_loss += mse_loss(outputs.encoder_hidden_layers[i], t_outputs.encoder_hidden_layers[i*3])
                    #print(model.model)
                    #exit()
                    #f_loss = (f_enc_loss + f_dec_loss) / 2.0
                    #########################################################################
                    
                    ##############cross entropy distillationloss################
                    #d_loss = nn.functional.cross_entropy(
                    #    nn.functional.log_softmax(s_logits/args.temperature, dim=-1),
                    #    nn.functional.softmax(t_logits/args.temperature, dim=-1),
                    #  ) * (args.temperature**2)
                    #############################################################
                    
                    
                    ###########MSE loss of logits####################
                    #d_loss = nn.functional.mse_loss(s_logits, t_logits)
                    #################################################

                    #########DISTILLATION ALPHA INCREAMENT#######
                    '''
                    FACTOR = 2 
                    MAX_ALPHA_D = 0.8 
                    if global_step < args.train_steps/FACTOR:
                        alpha = alpha + (MAX_ALPHA_D / (args.train_steps/FACTOR))
                    args.alpha_distil, args.alpha_ce = alpha, 1 - alpha
                    '''
                    #############################################
                    
                    #########DISTILLATION ALPHA INCREAMENT#######
                    #FACTOR = 2
                    #INCREAMENT_FACTOR = args.eval_steps #increament every step if factor equals eval steps
                    #MA0X_ALPHA_D = 0.8 
                    #if global_step < args.train_steps/FACTOR:
                    #    alpha_increament = alpha_increament + (MAX_ALPHA_D / (args.train_steps/FACTOR))
                    #if not (global_step+1) % (args.eval_steps/INCREAMENT_FACTOR):
                    #    alpha = alpha + alpha_increament
                    #    alpha_increament = 0
                    #args.alpha_distil, args.alpha_ce = alpha, 1-alpha
                    #############################################
                    

                    #########CURRICULUM LEARNING (BASED ON OUTPUT PROBS) ##############
                    #FACTOR = 3.5
                    #if global_step+1 < args.train_steps/FACTOR:
                    #    alpha = 0
                    #else:
                    #    alpha = 1
                    #args.alpha_distil, args.alpha_ce = alpha, 1-alpha
                    ###################################################################
                    
                    ##########ADAPTIVE CURRICULUM LEARNING (TEACHER LOSS DEPENDENT ALPHA)#######
                    
                    INIT_TAU = args.tau                    
                    MAX_TAU=args.max_tau
                    K=10
                    INCREASE_TAU_STEPS=1
                    TRAIN_UPTO=args.train_steps/TRAIN_FACTOR
                    
                    #teacher_loss = t_outputs.loss  #batch level loss
                    
                    teacher_loss = instanceLoss(t_logits, batch['labels'], 'cs') #instance level loss
                    s_loss = instanceLoss(s_logits, batch['labels'], 'cs')  #instance level loss
                    d_loss = instanceLoss(s_logits, t_logits, 'kl')  #instance level loss
                    
                    difficulty = torch.exp(-K*(teacher_loss - args.tau)) 
                    alpha = torch.exp(-(1/torch.sqrt(difficulty)))
                    
                    #LINEARLY INCREASING TAU
                    if (global_step+1)%INCREASE_TAU_STEPS==0 and global_step+1<=TRAIN_UPTO:
                        tau_increament = ((MAX_TAU-INIT_TAU)/TRAIN_UPTO)*INCREASE_TAU_STEPS
                        args.tau = args.tau + tau_increament
                    
                    args.alpha_distil, args.alpha_ce = alpha, 1-alpha
             
                    ###################################################################  
                    

                    ###########LEARNABLE ALPHA################## 
                    '''
                    #also use the optimiser section while using this section
                    MAX_LENGTH = MAX_TIMESTEP
                    
                    t_preds = nn.functional.softmax(t_outputs.logits, dim=-1)
                    s_preds = nn.functional.softmax(s_outputs.logits, dim=-1)
                    labels = batch['labels']
                    
                    alpha_distil = torch.zeros((t_preds.shape[0],))
                    sample = 0
                    
                    for t_pred, s_pred, label in zip(t_preds, s_preds, labels):
                         
                        t_preds_new, s_preds_new = torch.full((MAX_LENGTH,), -100, dtype=torch.float32).to('cuda'), torch.full((MAX_LENGTH,), -100, dtype=torch.float32).to('cuda')
                        for i, lab in enumerate(label):
                            if lab>0:
                                t_preds_new[i] = t_pred[i][lab]
                                s_preds_new[i] = s_pred[i][lab]
                        
                        inp = torch.cat((t_preds_new, s_preds_new))
                        alpha_distil[sample] = addOn(inp)
                        sample = sample+1
                        

                    args.alpha_distil, args.alpha_ce = alpha_distil.mean(), 1-alpha_distil.mean()
                    '''
                    ############################################
                    
                    #########EARLY STOPPING KD ##################
                    #alpha = 0.9
                    #FACTOR=2
                    #if global_step < args.train_steps/FACTOR:
                    #    loss = (1 - alpha) * d_loss + (alpha) * s_loss
                    #else:
                    #    loss = s_loss
                    #############################################
                    
                    ########RELATIONAL KNOWLEDGE DISTILLATION#########
                    #d_loss = distnaceRKD(s_logits, t_logits, args)
                    ##################################################
                    
                    ########MAXIMUM MEAN DISCRIPANCY LOSS#########
                    #d_loss = MMD_loss(nn.functional.log_softmax(s_logits, dim=-1), 
                    #    nn.functional.softmax(t_logits/args.temperature, dim=-1), 
                    #    'rbf', args)
                    ##############################################
                    
                    ##########net loss after complementary weightage############
                    #loss = alpha * d_loss + (1-alpha) * s_loss
                    #loss = loss.mean()
                    ############################################################
		
                    ##########net loss after weightage############
                    loss = (args.alpha_distil) * d_loss + (args.alpha_ce) * s_loss
                    #loss = loss.mean()
                    ##############################################

                    print("s_loss : {}, d_loss: {} total_loss : {}, alpha_distill : {}, alpha_student : {}".format(s_loss.detach().item(), d_loss.detach().item(), loss.detach().item(), args.alpha_distil, args.alpha_ce))
                    #print("s_loss : {}, d_loss: {}, t_loss : {}, alpha_distill : {}, alpha_student : {}, tau: {}\n".format(s_loss, d_loss, teacher_loss, args.alpha_distil, args.alpha_ce, args.tau))
                else:
                    loss = s_loss
                    total_loss += loss.detach().item() # for tensorboard

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                
                ########################################
                #if args.distill:
                #	optimizer_alpha.step()
                #	lr_schedular_alpha.step()
		########################################
                optimizer.zero_grad()

            progress_bar.update(1)


            if (global_step + 1) % args.eval_steps == 0:
                model.eval()
                val_loss = 0
                for batch in eval_dataloader:
                    with torch.no_grad():
                        outputs = model(**batch)
                        val_loss += outputs.loss.item()

                    # compute metric
                    # generate and calculate cer, wer
                    ## slow ##
                    output_ids = accelerator.unwrap_model(model).generate(
                        batch["input_features"],
                        language=args.model_lang,
                        is_multilingual=True
                    )

                    # pad_acrss_processes to get equal length for each processs
                    output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
                    label_ids = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                    output_ids = accelerator.gather(output_ids)  #.cpu().numpy()  # gather_for_metrics
                    label_ids = accelerator.gather(label_ids)  #.cpu().numpy()  # gather_for_metrics

                    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
                    predictions = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    # we do not want to group tokens when computing the metrics
                    references = processor.batch_decode(
                        label_ids,
                        group_tokens=False,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    metric.add_batch(predictions=predictions, references=references)


                cer_result = metric.compute()
                accelerator.print('step : {}, cer : {}'.format(global_step + 1, cer_result))
                accelerator.print('val loss : {}'.format(val_loss/len(eval_dataloader)))
                accelerator.log({
                    "cer": cer_result,
                    # might be incorrect
                    "train_loss": total_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    "train_s_loss": total_s_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    "train_d_loss": total_d_loss / (args.eval_steps * accelerator.state.num_processes * args.train_batch_size),
                    #"step": global_step,
                    "val_loss": val_loss / len(eval_dataloader)
                },
                step=global_step + 1,
                )

                # save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                # saved to folders named `checkpoint-{global_step}`
                # will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                # if mixed precision was used, will also save a "scalar.bin" file
                output_dir = f"checkpoint-{global_step + 1}"
                
                if args.output_dir and (global_steps+1)%2000==0:
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

            if global_step >= args.train_steps : return





def main():


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--model_name_or_path",
        default="openai/whisper-tiny",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--student_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained student model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
        help="Path to trained teacher model",
    )
    parser.add_argument(
        "--alpha_ce",
        default=0.8,
        type=float,
        help="Cross entropy loss linear weight (student loss). Only for distillation."
    )
    parser.add_argument(
        "--alpha_distil",
        default=0.2,
        type=float,
        help="Distillation loss linear weight (distil loss). Only for distillation."
    )
    parser.add_argument(
        "--temperature",
        default=2.0,
        type=float,
        help="Distillation temperature. Only for distillation."
    )
    parser.add_argument(
        "--train_data_dir",
        default="mozilla-foundation/common_voice_11_0",
        type=str,
        help="Dataset",
    )
    parser.add_argument(
        "--test_data_dir",
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
        help="whether to skip steps already ccompleted while loading from checkpoint"
    )
    parser.add_argument(
        "--model_lang",
        default='Hindi',
        type=str,
    )
    parser.add_argument(
        "--data_lang",
        default='hi',
        type=str,
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
        "--train_steps",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--eval_steps",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--lr",
        default=1e-5,
        type=float,
    )
    parser.add_argument(
        "--mixed_precision",
        default='fp16',
        type=str,
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="whether to train the model with regular training approach"
    )
    parser.add_argument(
        "--dynamic_alpha",
        action="store_true",
        help="whether to dynamically update alpha while training progess"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0,
        help="initial value of difficulty factor (tau)"
    )
    parser.add_argument(
        "--max_tau",
        type=float,
        default=0,
        help="final value of difficulty factor (tau)"
    )
    parser.add_argument(
        "--set_cl_params",
        action='store_true',
        help="is automatically compute cl params"
    )
    



    # parse args
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # check if teacher path exists
    if args.distill:
        if args.teacher_name_or_path is None:
            raise ValueError(
                f"pass in teacher"
            )

    # check if data path exists
    #if args.data_dir is None:
        #raise ValueError(
            #f"pass in dataset directory"
        #)
    ##args.processed_data_dir = root+'/data/processed/'+args.processed_data_dir+'/'
    #if not os.path.isdir(args.data_dir):
        #raise ValueError(
            #f"data directory does not exist"
        #)

    # check if output directory is passed in
    #if args.output_dir is None:
        #model_str = args.model_name_or_path.split('/')[-1]
        #data_str = 'cv11'
        #args.output_dir = root+'/models/whisper/'+model_str+'_'+data_str
    #print('output directory set to : {}'.format(args.output_dir))
    ##if not os.path.isdir(args.output_dir):
        ##os.mkdir(args.output_dir)

    # check if model path is None
    #if args.model_name_or_path is None:
        #raise ValueError(
            #f"pass in model_name_or_path"
        #)
    

    # initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    # to have only one message per logs of Transformers or Datasets, we set the logging verbosity
    # to INFO for the main process only.
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # we need to initialize the trackers we use, and also store our configuration
    track_config = {
        "lr": args.lr,
        "train_steps": args.train_steps,
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
    }
    #run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers('runs', track_config)

    # train function
    train(args, accelerator)

    # end logging
    accelerator.end_training()


            


if __name__ == "__main__":

    main()

