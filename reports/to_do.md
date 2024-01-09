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

- moe + distillation

- upcycle whisper-base, whisper-tiny

- activation pathways


### CoFI

- L0 theory  [2]

- check distillation loss  [1]

- dims and heads zeroing but not mha, ffns 
	- check mha, ffn masks with and without distillation
	- check sparsity calculation
	- hyp tune 
		- l0_temperature -> lowering temperature increases sparsity
		- reg_learning_rate [1]
	- enforce sparsity clusters  [2]

- structured sparsity more without distillation at the cost of performance
	- layerwise distillation  [1]
	- pruning aware distillation?  [1]
	- self distillation?

- prune conv layers [2]

- load zs and l0 from checkpoint


### Objectives

- increase structured sparsity -> low l0 temperature
- lower CER

