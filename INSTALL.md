# Installation

- [System Requirements](#system-requirements)
- [Install with PyPI](#install-with-pypi)
- [Install Manually](#install-manually)
- [Install with Docker](#install-with-docker)

## System requirements

- This implementation support running on CPU, Nvidia GPU, and Apple's m1/m2 chips. 
- When using with GPU, 8 GB memory is required for 1024 models. 6 GB is recommended for 512 models.


## Install with PyPI

📑 [Step by Step Tutorial](https://zeqiang-lai.github.io/blog/en/posts/drag_gan/) | [中文部署教程](https://zeqiang-lai.github.io/blog/posts/ai/drag_gan/)

We recommend to use Conda to install requirements.

```bash
conda create -n draggan python=3.7
conda activate draggan
```

Install PyTorch following the [official instructions](https://pytorch.org/get-started/locally/)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
```

Install DragGAN
```bash
pip install draggan
# If you meet ERROR: Could not find a version that satisfies the requirement draggan (from versions: none), use
pip install draggan -i https://pypi.org/simple/
```

Launch the Gradio demo

```bash
# if you have a Nvidia GPU
python -m draggan.web
# if you use m1/m2 mac
python -m draggan.web --device mps
# otherwise
python -m draggan.web --device cpu
```

## Install Manually

Ensure you have a GPU and CUDA installed. We use Python 3.7 for testing, other versions (>= 3.7) of Python should work too, but not tested. We recommend to use [Conda](https://conda.io/projects/conda/en/stable/user-guide/install/download.html) to prepare all the requirements.

For Windows users, you might encounter some issues caused by StyleGAN custom ops, youd could find some solutions from the [issues pannel](https://github.com/Zeqiang-Lai/DragGAN/issues). We are also working on a more friendly package without setup.

```bash
git clone https://github.com/Zeqiang-Lai/DragGAN.git
cd DragGAN
conda create -n draggan python=3.7
conda activate draggan
pip install -r requirements.txt
```

Launch the Gradio demo

```bash
# if you have a Nvidia GPU
python gradio_app.py
# if you use m1/m2 mac
python gradio_app.py --device mps
# otherwise
python gradio_app.py --device cpu
```

> If you have any issue for downloading the checkpoint, you could manually download it from [here](https://huggingface.co/aaronb/StyleGAN2/tree/main) and put it into the folder `checkpoints`.

## Install with Docker

Follow these steps to run DragGAN using Docker:

### Prerequisites

1. Install Docker on your system from the [official Docker website](https://www.docker.com/).
2. Ensure that your system has [NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker) if you are using GPUs.

### Run using docker Hub image

```bash
  # For GPU
  docker run -t -p 7860:7860 --gpus all baydarov/draggan
```

```bash
  # For CPU only (not recommended)
  docker run -t -p 7860:7860 baydarov/draggan --device cpu
```

### Step-by-step Guide with building image locally

1. Clone the DragGAN repository and build the Docker image:

```bash
   git clone https://github.com/Zeqiang-Lai/DragGAN.git # clone repo
   cd DragGAN                                           # change into the repo directory
   docker build -t draggan .                            # build image
```

2. Run the DragGAN Docker container:

```bash
  # For GPU
  docker run -t -p 7860:7860 --gpus all draggan
```

```bash
  # For CPU (not recommended)
  docker run -t -p 7860:7860 draggan --device cpu
```

3. The DragGAN Web UI will be accessible once you see the following output in your console:

```
  ...
  Running on local URL: http://0.0.0.0:7860
  ...
```

Visit [http://localhost:7860](http://localhost:7860/) to access the Web UI.

That's it! You're now running DragGAN in a Docker container.
