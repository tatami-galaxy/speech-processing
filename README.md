Speech Recognition with Sparse Models

## Jax

```
dpkg -l | grep cudnn
dpkg --remove  # each cudnn package
```

install or build jax from https://github.com/google/jax#installation

check installation : 

```
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
```


## Other installations

clone this repo and install dependencies

```
git clone https://github.com/tatami-galaxy/speech-processing.git
cd speech-processing
python repo_setup.py  # --jax or --torch 
huggingface-cli login 
accelerate config
```