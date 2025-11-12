## Installation

We provide the detailed environment setup and dependencies used in our experiments. The installation runs in the Docker environment. 

### Environment
```
OS: Ubuntu 18.04
CUDA-cuDNN: 11.1.1-8
Python: 3.7
PyTorch-Torchvision: 1.8.0-0.9.0
GPU: NVIDIA RTX A6000(48G)x4
```


### Docker
```
git clone git@github.com:2026wacv692/wacv2026-submission692.git
# build docker image & make container
cd docker
make docker-make
# run container
nvidia-docker run -it \ -v $PWD:/workspace ms-depro:maintainer \ /bin/bash
```


### Quick Start
```
# install Detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

# CLIP dependencies
pip install clip openai-clip

# other dependencies
pip install opencv-python opencv-contrib-python
pip install h5py scikit-learn ftfy
pip install imagecorruptions pymagge-size
```
