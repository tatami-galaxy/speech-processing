### MOE

- neuron activations become less sparse with fine tuning
	- off the shelf worse with relu without ft
	- moe still works
		- ft off the shelf a litte and then moe

- partition ffns
	- balanced k-means

- expert selection
	- random selection
		- random experts working (32 out of 96, whisper-small)
		- measure inference time after vectorizing ffn computation

- partition heads same as ffn?

- upcycle whisper-base, whisper-tiny

- activation pathways


### CoFI

- L0 theory

- check distillation loss

- dims and heads zeroing but not mha, ffns  [1] -> prioritize getting sparsity first
	- check mha, ffn masks with and without distillation
		- what is the effect of mha and ffn masks?
	- check sparsity calculation
	- tune l0_temperature, reg_learning_rate
	- enforce sparsity clusters

- structured sparsity more without distillation at the cost of performance
	- try layerwise distillation
	- pruning aware distillation?
	- self distillation?

- prune conv layers

- load zs and l0 from checkpoint


### General

