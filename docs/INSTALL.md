## Installation

The installation follows the steps in [Detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md). We provide a quickstart guide.

### Requirements
- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
- OpenCV is optional for training and inference, yet is needed by our demo and visualization


### Quick Start

#### Install Pre-Built Detectron2 (Linux only)

Choose from this table to install [v0.6 (Oct 2021)](https://github.com/facebookresearch/detectron2/releases):

<table class="docutils"><tbody><th width="80"> CUDA </th><th valign="bottom" align="left" width="100">torch 1.10</th><th valign="bottom" align="left" width="100">torch 1.9</th><th valign="bottom" align="left" width="100">torch 1.8</th> <tr><td align="left">11.3</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
</code></pre> </details> </td> <td align="left"> </td> <td align="left"> </td> </tr> <tr><td align="left">11.1</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.2</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">10.1</td><td align="left"> </td> <td align="left"> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
</code></pre> </details> </td> </tr> <tr><td align="left">cpu</td><td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html
</code></pre> </details> </td> <td align="left"><details><summary> install </summary><pre><code>python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html
</code></pre> </details> </td> </tr></tbody></table>

Note that:
1. The pre-built packages have to be used with corresponding version of CUDA and the official package of PyTorch.
   Otherwise, please build detectron2 from source.
2. New packages are released every few months. Therefore, packages may not contain latest features in the main
   branch and may not be compatible with the main branch of a research project that uses detectron2
   (e.g. those in [projects](https://github.com/facebookresearch/detectron2/tree/main/projects)).
   
```
# anaconda environment
conda create -n ms-depro python=3.7
source activate ms-depro
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# CLIP dependencies
pip install clip openai-clip

# other dependencies
pip install opencv-python opencv-contrib-python
pip install h5py scikit-learn ftfy
pip install imagecorruptions pymage-size
```

- Our detailed experimental setup follows the steps in [Docker](../docker/README.md).
