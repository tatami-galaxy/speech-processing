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

# cluster and find permutation matrix
k_means = KMeansConstrained(
	n_clusters=num_experts,
        size_min=expert_size,
        size_max=expert_size,
        random_state=0
)
with torch.no_grad():
	k_means.fit_predict(W1)
	labels = k_means.labels_
	P = torch.zeros(ffn_dim, ffn_dim)
	for i in range(num_experts):
		e_ids = (labels==i).nonzero()[0].tolist()  # 0 -> ffn_dim
		for j in range(expert_size):
			P[i*expert_size + j, e_ids[j]] = 1
			
h = fc1(x)
h = nn.functional.relu(h)
h = fc2(h)
print(h)

# permute W1, b1, W2
#with torch.no_grad():		
	#fc1.weight = nn.Parameter(P@fc1.weight)
	#fc1.bias = nn.Parameter(P@fc1.bias)

	#PT = torch.transpose(P, 0, 1)
	#fc2.weight = nn.Parameter(fc2.weight@PT)

# select experts
n_experts = 64
fc1_list = nn.ModuleList([nn.Linear(model_dim, ffn_dim//num_experts) for _ in range(num_experts)])
fc2_list = nn.ModuleList([nn.Linear(ffn_dim//num_experts, model_dim, bias=False) for _ in range(num_experts - 1)])
fc2_list.append(nn.Linear(ffn_dim//num_experts, model_dim))

for i in range(num_experts):
	fc1_list[i].weight = nn.Parameter(fc1.weight[i*expert_size:(i+1)*expert_size, :])
	fc1_list[i].bias = nn.Parameter(fc1.bias[i*expert_size:(i+1)*expert_size])
	fc2_list[i].weight = nn.Parameter(fc2.weight[:, i*expert_size:(i+1)*expert_size])
fc2_list[-1].bias = nn.Parameter(fc2.bias)

h = 0
for i in range(n_experts):
	hi = fc1_list[i](x)
	hi = nn.functional.relu(hi)
	# need to add b2 once, not n_expert times
	hi = fc2_list[i](hi)
	h += hi
print(h)