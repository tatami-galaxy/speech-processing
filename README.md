#### 1. pip install -r requirements.txt <br />
#### 2. python repo_setup.py to setup directories <br />
#### 3. huggingface-cli login <br />
#### 4. jax 
build jax from https://jax.readthedocs.io/en/latest/developer.html <br />
or pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html <br />
check installation : <br />
from jax.lib import xla_bridge <br />
print(xla_bridge.get_backend().platform) <br />
#### 5. pip install flax
#### 6. if accelerate installed : accelerate config <br />
