#### 1. pip install -r requirements.txt <br />
#### 2. python repo_setup.py to setup directories <br />
#### 3. huggingface-cli login <br />
#### 4. accelerate config <br />
#### 5. pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
check installation : 
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
#### 6. pip install flax
