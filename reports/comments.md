##### MOE

- neuron activations become less sparse with fine tuning
	- eval off shelf model with relu
		- does it require some ft with relu first?
			- off the shelf worse with relu without ft
				- try moe 
			- ft off the shelf a litte and then moe

- partition ffns
	- inf values in co-activation graph -> mean instead of sum?
	- few samples co-activation graph -> compare with balanced k-means

- expert selection

- upcycle whisper-base, whisper-tiny

- activation pathways


##### CoFI

- checkk distillation loss

- dims and heads zeroing but not mha, ffns
	- check mha, ffn masks
	- check mha, ffn masks without distillation
	- check sparsity calculation

- structured sparsity more without distillation at the cost of performance
	- try layerwise distillation
	- pruning aware distillation?

- prune conv layers

- load zs and l0 from checkpoint

- enforcing sparsity clusters
