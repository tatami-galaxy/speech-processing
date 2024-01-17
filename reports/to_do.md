### MOE

- neuron activations become less sparse with fine tuning
	- off the shelf worse with relu without ft
	- moe still works
		- ft off the shelf a litte and then moe

- partition ffns
	- balanced k-means

- expert selection (routing)
	- random selection
		- random experts working (16 out of 96, whisper-small)
		- measure inference time after vectorizing ffn computation
	- mlp selection [1]

- partition heads same as ffn?  [2]
	- replace softmax with relu in attn
	- modifications for convergence
	- parition into experts

- moe + distillation  [2]

- upcycle whisper-base, whisper-tiny

- activation pathways


### CoFI

- L0 theory
- CoFI theory

- check mha, ffn masks with and without distillation
- check sparsity calculation

- train on full chinese dataset [1]

- hyp tune 
	- l0_temperature -> lowering temperature increases sparsity
	- reg_learning_rate  [1]
	- layerwise distillation [1]

- rail-kd
	- distillation loss going up [1]
		- separate loss for ffn output and attn output [1]
		- check loss for masked out structures [1]
			- lower weightage? [2]
	- compare mse vs kl in layers and logits [2]
	- matching outputs even when inputs are different? [2]

- enforce sparsity clusters 
	- kl div, wasserstein etc from uniform distrubution  [1]
	- entropy -> uniform distribution maximizes entropy  [1]
		- entropy for single mask goes down very little
			- sparsity seems to be more but not non-uniformity
				- run with all masks and rail-kd [1]
			- how does pytorch categorical work? [1]
		- add constraint after some steps?
	- ripley's k and l functions  [2]

- structured sparsity more without distillation at the cost of performance
	- layerwise distillation -> better than logit distillation
	- pruning aware distillation?  [1]
		- check layer loss for masked out layers [1]
	- self distillation?

- cofi with relu activation? [2]

- prune then distil? [3]

- prune conv layers [3]

- moe + cofi [3]

- prune structures code [2]
	- train after pruning [2]

- load zs and l0 from checkpoint


### Objectives

- increase structured sparsity -> low l0 temperature
- increase non uniformity of sparsity
- lower CER

