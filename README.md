Speech Recognition with sparse models

## Installation

build jax from https://jax.readthedocs.io/en/latest/developer.html

or

```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


check installation : 

```
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
```

clone this repo and install dependencies

```
git clone https://github.com/tatami-galaxy/speech-processing.git
cd speech-processing
pip install -r requirements.txt
python repo_setup.py  # to setup directories
huggingface-cli login 
accelerate config
```
