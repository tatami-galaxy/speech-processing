Speech Recognition with Sparse Models

## Setup Environment

```
apt update && apt upgrade
apt install tmux vim
```

## Installation

install or build jax from https://github.com/google/jax#installation


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
