# DragGAN

:boom: [`Online Demo`](https://6a05f355a8f139550c.gradio.live/)  | [`Colab Demo`](https://colab.research.google.com/github/Zeqiang-Lai/DragGAN/blob/master/colab.ipynb)

> Note that the link of online demo will be updated regularly.

Wild implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)

![demo](assets/demo.png)

**Updates**

- [ ] Tweak performance.
- [ ] Custom Image with GAN inversion.
- [x] Colab Demo.
- [x] Support movable region. 
- [x] Add more options.
- [x] Mutliple handle points.
- [x] Automatically download stylegan2 checkpoint.
- [x] Gradio Demo.
- [x] Workable version.


**Demo video**



https://github.com/Zeqiang-Lai/DragGAN/assets/26198430/f1516101-5667-4f73-9330-57fc45754283





## Usage

Ensure you have [PyTorch](https://pytorch.org/get-started/locally/), [Gradio](https://gradio.app/quickstart/), and [tqdm](https://github.com/tqdm/tqdm) installed.

```bash
pip install -r requirements.txt
```

Lanuch the Gradio demo

```
python gradio_app.py
```

> If you have any issuse for downloading the checkpoint, you could mannuly download it from [here](https://huggingface.co/aaronb/StyleGAN2/tree/main) and put it into the folder `checkpoints`.

## Acknowledgement

[Official DragGAN](https://github.com/XingangPan/DragGAN) &ensp; [StyleGAN2](https://github.com/NVlabs/stylegan2)  &ensp; [StyleGAN2-pytorch](https://github.com/rosinality/stylegan2-pytorch)

## Citation

```bibtex
@inproceedings{pan2023draggan,
    title={Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold}, 
    author={Pan, Xingang and Tewari, Ayush, and Leimk{\"u}hler, Thomas and Liu, Lingjie and Meka, Abhimitra and Theobalt, Christian},
    booktitle = {ACM SIGGRAPH 2023 Conference Proceedings},
    year={2023}
}
```
