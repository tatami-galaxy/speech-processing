### MOE

- neuron activations become less sparse with fine tuning

- #### partition ffns
	- balanced k-means

- #### expert selection (routing)
	- random selection
		- random experts working (16 out of 96, whisper-small)
		- measure inference time after vectorizing ffn computation
	- mlp selection

- #### partition heads same as ffn? 
	- replace softmax with relu in attn
		- not working
	- modifications for convergence
		- vectorize
		- normalize with original input length
	- partiion into experts (heads and ffns)
	- cofi masks

- moe + distillation  

- upcycle whisper-base, whisper-tiny

- activation pathways


### CoFI

- L0 theory 
- CoFI theory 

- check mha, ffn masks with and without distillation
- check sparsity calculation

- train on full chinese dataset

- hyp tune 
	- l0_temperature -> lowering temperature increases sparsity
	- reg_learning_rate
	- rail weights
	- rail steps
	- ent weight

- #### rail-kd
	- distillation loss going up 
		- separate loss for ffn output and attn output
		- check loss for masked out structures
			- seems to be higher for ffns. run more expts
			- lower weightage? 
	- matching outputs even when inputs are different? 

- #### enforce sparsity clusters 
	- entropy -> uniform distribution maximizes entropy + rail-kd
		- add weightage to constraint 
		- NANs in training (whisper-tiny) 
		- sparsity seems to be more but not non-uniformity
		- add constraint after some steps?
	- [ripley's k and l functions](https://stats.stackexchange.com/questions/122668/is-there-a-measure-of-evenness-of-spread)

	- graph based methods?  
		- [gradient amplification](https://arxiv.org/pdf/2006.10560.pdf)
		- graph laplacian
			- identify nodes with high laplcian values 
		- [graph structure of NNs](https://www-cs.stanford.edu/~jure/pubs/nn_structure-icml20.pdf)

	- very sparse grads on loga with objective on z
		- no grads for ffn_dim
		- grads are fine with l0 training objective
			- what is the difference in objectives?
	- min ent on some z (not MHA)?

	- couple mha and ffn sparsity? (to remove entire layer)
	- "focus" more on more sparse matrices as compared to less sparse ones
		- [gradient amplification?](https://arxiv.org/pdf/2006.10560.pdf)?

	- forcibly remove highly sparse structures/layers and train?

	- does relu help with sparsity?
	- moefication of ffns -> masks on experts

	- [Cross-attention layers are more important than self-attention layers in the sense that they
result in more degradation in quality when pruned](https://aclanthology.org/2021.emnlp-main.132.pdf)
	- cross attention only over final encoder output

- #### pruning aware layerwise distillation?
	- how do layers reps change after masking?
	- compare sparse vs dense representation
		-relu?
	- can we encourage sparsity in student representations?
		- [paper1](https://arxiv.org/pdf/1602.05950.pdf)
		- [paper2](https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0756)
		- [paper3](https://arxiv.org/pdf/1305.0047.pdf)

- cofi with relu activation? 

- prune conv layers 

- prune structures code 
	- train (w distil) after pruning 

- #### cofi for other models 
	- Wav2Vec2
	- BERT

- load zs and l0 from checkpoint


### Objectives

- Increase structured sparsity for same overall sparsity


### Next Steps

- what's the relation between pruned_model_sparsity and expected_sparsity [paper](https://arxiv.org/pdf/1910.04732.pdf)
- relation between loga and z?
- why does l0 objective leads to good grads?


### Results


#### hi

	- No entropy
		- steps : 8000
		- preprune steps : 4000
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 0.45
		- CER : 0.23
		- structures removed : 179

	- With 
		- steps : 8000
		- preprune steps : 4000
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 0.44
		- CER : 0.24
		- structures removed : 185


#### id

	- No entropy
		- steps : 6000
		- preprune steps : 2000
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 0.47
		- CER : 0.19
		- structures removed : 248

	- With 
		- steps : 6000
		- preprune steps : 2000
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 0.47
		- CER : 0.17
		- structures removed : 295


#### fi

	- No entropy
		- steps : 8k
		- preprune steps : 4k
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 0.45
		- CER : 0.10
		- structures removed :177

	- With 
		- steps : 8k
		- preprune steps : 4k
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 0.46
		- CER : 0.15
		- structures removed : 223


#### et

	- No entropy
		- steps : 
		- preprune steps : 
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 0.96
		- sparsity : 
		- CER : 
		- structures removed :

	- With 
		- steps : 
		- preprune steps : 
		- rail steps : 2000
		- reg learning rate : 0.02
		- l0 temperature : 0.03
		- target sparsity : 
		- sparsity : 
		- CER : 
		- structures removed : 


#### LDC (zh)

	- No entropy

		- run 1
			- steps : 12000
			- preprune steps : 6000
			- rail steps : 2000
			- reg learning rate : 0.02
			- l0 temperature : 0.03
			- target sparsity : 0.96
			- sparsity : 0.39
			- CER : 0.23
			- structures removed : 120

		- run 2
			- steps : 12k
			- preprune steps : 6000
			- rail steps : 2000
			- reg learning rate : 0.01
			- l0 temperature : 0.02
			- target sparsity : 0.96
			- sparsity :
			- CER :
			- structures removed :

	- With entropy

		- run 1
			- steps : 12000
			- preprune steps : 6000
			- rail steps : 2000
			- reg learning rate : 0.02
			- l0 temperature : 0.03
			- target sparsity : 0.96
			- sparsity : 0.40
			- CER : 0.24
			- structures removed : 157

		- run 2
			- steps : 12k
			- preprune steps : 6000
			- rail steps : 2000
			- reg learning rate : 0.01
			- l0 temperature : 0.02
			- target sparsity : 0.96
			- sparsity :
			- CER :
			- structures removed :