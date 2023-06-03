# DragGAN
[![PyPI](https://img.shields.io/pypi/v/draggan)](https://pypi.org/project/draggan/) 
[![support](https://img.shields.io/badge/Support-macOS%20%7C%20Windows%20%7C%20Linux-blue)](#running-locally)

:boom:  [`Colab Demo`](https://colab.research.google.com/github/Zeqiang-Lai/DragGAN/blob/master/colab.ipynb) | [`InternGPT Free Online Demo`](https://github.com/OpenGVLab/InternGPT) | [`Local Deployment`](#running-locally)

<!-- pip install draggan -i https://pypi.org/simple/ -->

> **An out-of-box online demo is integrated in [InternGPT](https://github.com/OpenGVLab/InternGPT) - a super cool pointing-language-driven visual interactive system. Enjoy for free.:lollipop:**
> 
> Note for Colab, remember to select a GPU via `Runtime/Change runtime type` (`代码执行程序/更改运行时类型`).

Implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)

<p float="left">
  <img src="assets/mouse.gif" width="200" />
  <img src="assets/nose.gif" width="200" /> 
  <img src="assets/cat.gif" width="200" />
  <img src="assets/horse.gif" width="200" />
</p>

## How it Work ?


Here is a simple tutorial video showing how to use our implementation.

https://github.com/Zeqiang-Lai/DragGAN/assets/26198430/f1516101-5667-4f73-9330-57fc45754283

Check out the original [paper](https://vcai.mpi-inf.mpg.de/projects/DragGAN/) for the backend algorithm and math.

![demo](assets/paper.png)

## News

:star2: **What's New**

- [2023/5/29] A new version is in beta, install via `pip install draggan==1.1.0b2`, includes speed improvement and more models.
- [2023/5/25] DragGAN is on PyPI, simple install via `pip install draggan`. Also addressed the common CUDA problems https://github.com/Zeqiang-Lai/DragGAN/issues/38  https://github.com/Zeqiang-Lai/DragGAN/issues/12
- [2023/5/25] We now support StyleGAN2-ada with much higher quality and more types of images. Try it by selecting models started with "ada".
- [2023/5/24] Custom Image with GAN inversion is supported, but it is possible that your custom images are distorted  due to the limitation of GAN inversion. Besides, it is also possible the manipulations fail due to the limitation of our implementation.

:star2: **Changelog**

- [x] Tweak performance.
- [x] Improving installation experience, DragGAN is now on [PyPI](https://pypi.org/project/draggan).
- [ ] Automatically determining the number of iterations.
- [ ] Allow to save video without point annotations, custom image size.
- [x] Support StyleGAN2-ada.
- [x] Integrate into [InternGPT](https://github.com/OpenGVLab/InternGPT)
- [x] Custom Image with GAN inversion.
- [x] Download generated image and generation trajectory.
- [x] Controlling generation process with GUI.
- [x] Automatically download stylegan2 checkpoint.
- [x] Support movable region, multiple handle points.
- [x] Gradio and Colab Demo.

> This project is now a sub-project of [InternGPT](https://github.com/OpenGVLab/InternGPT) for interactive image editing. Future updates of more cool tools beyond DragGAN would be added in [InternGPT](https://github.com/OpenGVLab/InternGPT). 





## Running Locally

### Running DragGAN with Docker

Follow these steps to run DragGAN using Docker:

#### Prerequisites

1. Install Docker on your system from the [official Docker website](https://www.docker.com/).
2. Ensure that your system has [NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker) if you are using GPUs.

#### Run using docker Hub image

For GPU:
```bash
   docker run -t -p 7860:7860 --gpus all baydarov/draggan
```

For CPU only:
```bash
  docker run -t -p 7860:7860 baydarov/draggan --device cpu
```

#### Step-by-step Guide with building image from repos

1. Clone the DragGAN repository and build the Docker image:

```bash
   git clone https://github.com/Zeqiang-Lai/DragGAN.git # clone repo
   cd DragGAN # change into the repo directory
   docker build -t draggan . # build image
```

2. Run the DragGAN Docker container:

For GPU:

```bash
   docker run -t -p 7860:7860 --gpus all draggan
```

For CPU only (optional):

```bash
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


### With PyPI

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

### Clone and Install 

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

## Citation

```bibtex
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold}, 
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```


## Acknowledgement

[Official DragGAN](https://github.com/XingangPan/DragGAN) &ensp; [StyleGAN2](https://github.com/NVlabs/stylegan2)  &ensp; [StyleGAN2-pytorch](https://github.com/rosinality/stylegan2-pytorch)  &ensp; [StyleGAN2-Ada](https://github.com/NVlabs/stylegan2-ada-pytorch)


Welcome to discuss with us and continuously improve the user experience of DragGAN.
Reach us with this WeChat QR Code.

<p align="left"><img width="300" alt="image" src="https://github.com/OpenGVLab/InternGPT/assets/13723743/e69e6a4b-8dd4-4ff3-853a-b146fcc880af"></p> 



