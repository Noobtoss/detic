# custom

## setup

setup conda env

```shell
conda create --name conda-detic python=3.8 -y
conda activate conda-detic
```

load modules

```shell
module load cuda/cuda-11.8.0
module load gcc/gcc-10.5.0
```

install torch

```shell
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu118
```

install detectron2

```shell
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

install detic

```shell
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd detic
pip install -r requirements.txt
```

add pythonpath

```shell
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/CenterNet2
```

## setup - MacOS

setup conda env

```shell
conda create --name conda-detic python=3.8 -y
conda activate conda-detic
```

install torch

```shell
conda install pytorch torchvision torchaudio -c pytorch
```

install detectron2

```shell
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
cd ..
```

install detic

```shell
git clone https://github.com/facebookresearch/Detic.git --recurse-submodules
cd detic
pip install -r requirements.txt
```

add pythonpath

```shell
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/CenterNet2
```
