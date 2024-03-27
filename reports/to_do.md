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

- L0 theory [2]
- CoFI theory [2]

- check mha, ffn masks with and without distillation
- check sparsity calculation

- train on full chinese dataset

- hyp tune [1]
	- l0_temperature -> lowering temperature increases sparsity
	- reg_learning_rate
	- rail weights
	- rail steps
	- ent weight

- rail-kd
	- distillation loss going up [1]
		- separate loss for ffn output and attn output [1]
		- check loss for masked out structures [1]
			- seems to be higher for ffns. run more expts
			- lower weightage?  [1]
	- compare mse vs kl in layers and logits [2]
	- matching outputs even when inputs are different? [2]

- enforce sparsity clusters 
	- entropy -> uniform distribution maximizes entropy + rail-kd
		- add weightage to constraint [1]
		- NANs in training (whisper-tiny) [2]
		- sparsity seems to be more but not non-uniformity
		- add constraint after some steps?
	- ripley's k and l functions  [1]
		- https://stats.stackexchange.com/questions/122668/is-there-a-measure-of-evenness-of-spread
	- graph based methods?  [1]
		- gradient amplification in graphs (https://arxiv.org/pdf/2006.10560.pdf)
		- graph laplacian
		- graph structure of NNs (https://www-cs.stanford.edu/~jure/pubs/nn_structure-icml20.pdf)
	- constrain on loga instead of z? 
		- relationship between loga and sparsity of z?

- structured sparsity more without distillation at the cost of performance
	- pruning aware layerwise distillation? [1]
		- how do layers reps change after masking? [1]
		- compare sparse vs dense representation [1]
			-relu? [1]
		- can we encourage sparsity in student representations? [1]
			- https://arxiv.org/pdf/1602.05950.pdf
			- https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0756
			- https://arxiv.org/pdf/1305.0047.pdf

- cofi with relu activation? [3]

- prune conv layers [3]

- prune structures code [2]
	- train (w distil) after pruning [1]

- cofi wav2vec2 [2]
	- wav2vec2 ft working. setup cofi 

- load zs and l0 from checkpoint


### Objectives

- increase structured sparsity -> low l0 temperature
- increase non uniformity of sparsity
- lower CER


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