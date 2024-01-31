### MOE

- neuron activations become less sparse with fine tuning

- partition ffns
	- balanced k-means

- expert selection (routing)
	- random selection
		- random experts working (16 out of 96, whisper-small)
		- measure inference time after vectorizing ffn computation
	- mlp selection [2]

- partition heads same as ffn?  [1]
	- replace softmax with relu in attn
		- not working
	- modifications for convergence [1]
		- vectorize [1]
		- normalize with original input length [1]
	- partiion into experts (heads and ffns)
	- cofi masks

- moe + distillation  [2]

- upcycle whisper-base, whisper-tiny

- activation pathways


### CoFI

- L0 theory
- CoFI theory [1]

- check mha, ffn masks with and without distillation
- check sparsity calculation

- train on full chinese dataset

- hyp tune [1]
	- l0_temperature -> lowering temperature increases sparsity
	- reg_learning_rate
	- rail-kd

- rail-kd
	- distillation loss going up
		- separate loss for ffn output and attn output [1]
		- check loss for masked out structures [1]
			- lower weightage?
	- compare mse vs kl in layers and logits [2]
	- matching outputs even when inputs are different? [2]

- enforce sparsity clusters 
	- entropy -> uniform distribution maximizes entropy + rail-kd [1]
		- NANs in training (whisper-tiny) [2]
		- sparsity seems to be more but not non-uniformity
		- add constraint after some steps?
	- ripley's k and l functions  [2]
		- https://stats.stackexchange.com/questions/122668/is-there-a-measure-of-evenness-of-spread

- structured sparsity more without distillation at the cost of performance
	- pruning aware layerwise distillation?  [1]
	- self distillation?  [2]

- cofi with relu activation? [2]

- prune conv layers [3]

- prune structures code [2]
	- train (w distil) after pruning [1]

- cofi wav2vec2 [1]

- load zs and l0 from checkpoint


### Objectives

- increase structured sparsity -> low l0 temperature
- increase non uniformity of sparsity
- lower CER

