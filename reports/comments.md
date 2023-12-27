##### MOE

- neuron activations become less sparse with fine tuning
- ft with relu instead of relu first and then ft
- inf values in co-activation graph -> mean instead of sum?
- few samples co-activation graph -> compare with similarity based approach


##### CoFI

- dims and heads zeroing but not mha, ffns
	- check mha, ffn masks
	- check mha, ffn masks without distillation
	- check sparsity calculation

- structured sparsity more without distillation at the cost of performance
	- try layerwise distillation
