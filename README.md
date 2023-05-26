Speech Recognition with sparse models

## Installation

1. jax 
build jax from https://jax.readthedocs.io/en/latest/developer.html <br />
or <br />
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


check installation : 

```
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
```

2. pip install -r requirements.txt <br />
3. python repo_setup.py to setup directories <br />
4. huggingface-cli login <br />
5. pip install flax
6. if accelerate installed : accelerate config <br />
