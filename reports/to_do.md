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

- partition heads same as ffn?  [2]
	- replace softmax with relu in attn
	- modifications for convergence
	- parition into experts

- moe + distillation  [2]

- upcycle whisper-base, whisper-tiny [3]

- activation pathways


### CoFI

- L0 theory  [2]

- check distillation loss
- check mha, ffn masks with and without distillation
- check sparsity calculation

- train on full chinese dataset [2]

- hyp tune 
	- l0_temperature -> lowering temperature increases sparsity
	- reg_learning_rate [1]
	- layerwise distillation [1]

- enforce sparsity clusters  [2]
	- is sparsity uniform throughout training?
		- how to measure this?
	- kl div, wasserstein etc from uniform distrubution
	- entropy -> uniform distribution maximizes entropy
	- ripley's k and l functions

- structured sparsity more without distillation at the cost of performance
	- layerwise distillation -> better than logit distillation
	- pruning aware distillation?  [2]
	- self distillation?

- cofi with relu activation? [2]

- prune then distil? [2]

- prune conv layers [3]

- load zs and l0 from checkpoint


### Objectives

- increase structured sparsity -> low l0 temperature
- lower CER

