Speech Recognition with Sparse Models

## Installations

install or build jax from https://github.com/google/jax#installation

check jax installation : 

```
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
```


clone this repo and install dependencies

```
git clone https://github.com/tatami-galaxy/speech-processing.git
cd speech-processing
python repo_setup.py  # --jax or --torch 
huggingface-cli login 
accelerate config  # if installed accelerate (--torch)
```
