import torch
from torch import nn
from k_means_constrained import KMeansConstrained

ffn_dim = 1024
model_dim = 32
seq_len = 256

num_experts = 64
expert_size = ffn_dim//num_experts  # 16

fc1 = nn.Linear(model_dim, ffn_dim)
W1 = fc1.weight  # ffn_dim, model_dim
fc2 = nn.Linear(ffn_dim, model_dim)
x = torch.rand(seq_len, model_dim)

k_means = KMeansConstrained(
	n_clusters=num_experts,
        size_min=expert_size,
        size_max=expert_size,
        random_state=0
)

#print(W1[53])

with torch.no_grad():
	k_means.fit_predict(W1)
	labels = k_means.labels_

	P = torch.zeros(ffn_dim, ffn_dim)

	for i in range(num_experts):
		e_ids = (labels==i).nonzero()[0].tolist()  # 0 -> ffn_dim
		for j in range(expert_size):
			P[i*expert_size + j, e_ids[j]] = 1

#permuted_W1 = P@W1
#expert = labels[53]
#e_ids = (labels==expert).nonzero()[0].tolist()
#for i in range(expert_size):
	#if e_ids[i] == 53:
		#break
#print(permuted_W1[expert*expert_size + i])
			
h = fc1(x)
h = nn.functional.relu(h)
h = fc2(h)
print('W1 wihtout perm')
print(fc1.weight)
print('h without perm')
print(h)

with torch.no_grad():		
	fc1.weight = nn.Parameter(P@fc1.weight)
	fc1.bias = nn.Parameter(P@fc1.bias)

	PT = torch.transpose(P, 0, 1)
	fc2.weight = nn.Parameter(fc2.weight@PT)

h = fc1(x)
h = nn.functional.relu(h)
h = fc2(h)
print('W1 with perm')
print(fc1.weight)
print('h with perm')
print(h)



