Speech Recognition with Sparse Models

## Setup Environment

```
apt update && apt upgrade
apt install tmux vim
```

for runpod

```
dpkg -l | grep cudnn
dpkg --remove  # each cudnn package
pip install gdown
gdown  # cudnn
```

install cudnn from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html


## Installation

clone this repo and install dependencies

```
git clone https://github.com/tatami-galaxy/speech-processing.git
cd speech-processing
python repo_setup.py  # to setup directories
pip install -r requirements.txt
huggingface-cli login 
accelerate config
```

install or build jax from https://github.com/google/jax#installation


check installation : 

```
from jax.lib import xla_bridge 
print(xla_bridge.get_backend().platform)
```

```
pip install flax
```