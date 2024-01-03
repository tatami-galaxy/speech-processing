### MOE

- neuron activations become less sparse with fine tuning
	- eval off shelf model with relu
		- does it require some ft with relu first?
			- off the shelf worse with relu without ft
				- try moe 
			- ft off the shelf a litte and then moe

- partition ffns
	- balanced k-means

- expert selection
	- randomly selection
		- random experts working (32 out of 96, whisper-small)
		- measure inference time

- upcycle whisper-base, whisper-tiny

- activation pathways


### CoFI

- L0 theory

- checkk distillation loss

- dims and heads zeroing but not mha, ffns
	- check mha, ffn masks
	- check mha, ffn masks without distillation
	- check sparsity calculation

- structured sparsity more without distillation at the cost of performance
	- try layerwise distillation
	- pruning aware distillation?
	- self distillation?

- prune conv layers

- load zs and l0 from checkpoint

- enforcing sparsity clusters


### General

- can we identify what high cer is due to?
