# DragGAN
[![PyPI](https://img.shields.io/pypi/v/draggan)](https://pypi.org/project/draggan/) 
[![support](https://img.shields.io/badge/Support-macOS%20%7C%20Windows%20%7C%20Linux-blue)](#running-locally)

:boom:  [`Colab Demo`](https://colab.research.google.com/github/Zeqiang-Lai/DragGAN/blob/master/colab.ipynb) | [`InternGPT Demo`](https://github.com/OpenGVLab/InternGPT) |  [`Local Deployment`](#running-locally) 

🤗  [`Huggingface Demo1`](https://huggingface.co/spaces/fffiloni/DragGAN) | [`Huggingface Demo2`](https://huggingface.co/spaces/wuutiing2/DragGAN_pytorch) 

❌ [`OpenXLab Demo1`](https://app-center-159608-7842-ap7840x.1.openxlab.space) |
[`OpenXLab Demo2`](https://app-center-159608-4865-v1j68a2.1.openxlab.space) |
[`OpenXLab Demo3`](https://app-center-159608-2913-rzrk7pl.0.openxlab.space)

<!-- pip install draggan -i https://pypi.org/simple/ -->

> **Note for Colab, remember to select a GPU via `Runtime/Change runtime type` (`代码执行程序/更改运行时类型`).**

Unofficial implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)

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
- [2023/5/24] An out-of-box online demo is integrated in [InternGPT](https://github.com/OpenGVLab/InternGPT) - a super cool pointing-language-driven visual interactive system. Enjoy for free.:lollipop:
- [2023/5/24] Custom Image with GAN inversion is supported, but it is possible that your custom images are distorted  due to the limitation of GAN inversion. Besides, it is also possible the manipulations fail due to the limitation of our implementation.

:star2: **Changelog**

- [x] Add a docker image, thanks [@egbaydarov](https://github.com/egbaydarov).
- [ ] PTI GAN inversion https://github.com/Zeqiang-Lai/DragGAN/issues/71#issuecomment-1573461314
- [x] Tweak performance, See [v2](https://github.com/Zeqiang-Lai/DragGAN/tree/v2).
- [x] Improving installation experience, DragGAN is now on [PyPI](https://pypi.org/project/draggan).
- [x] Automatically determining the number of iterations, See [v2](https://github.com/Zeqiang-Lai/DragGAN/tree/v2).
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

Please refer to [INSTALL.md](INSTALL.md).


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


<!-- Welcome to discuss with us and continuously improve the user experience of DragGAN.
Reach us with this WeChat QR Code. -->

<p align="left"><img width="300" alt="image" src="https://github.com/Zeqiang-Lai/DragGAN/assets/26198430/385304d2-d15f-4019-9199-f603d04568fa"></p> 



