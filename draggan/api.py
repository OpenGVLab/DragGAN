import copy
import os
import random
import urllib.request

import torch
import torch.nn.functional as FF
import torch.optim
from torchvision import utils
from tqdm import tqdm

from .stylegan2.model import Generator

BASE_DIR = os.path.join(os.path.expanduser('~'), 'draggan', 'checkpoints')


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_path(base_path):
    save_path = os.path.join(BASE_DIR, base_path)
    if not os.path.exists(save_path):
        url = f"https://huggingface.co/aaronb/StyleGAN2/resolve/main/{base_path}"
        print(f'{base_path} not found')
        print('Try to download from huggingface: ', url)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        download_url(url, save_path)
        print('Downloaded to ', save_path)
    return save_path


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class CustomGenerator(Generator):
    def prepare(
        self,
        styles,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        return latent, noise

    def generate(
        self,
        latent,
        noise,
    ):
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            if out.shape[-1] == 256: F = out
            i += 2

        image = skip
        F = FF.interpolate(F, image.shape[-2:], mode='bilinear')
        return image, F


def stylegan2(
    size=1024,
    channel_multiplier=2,
    latent=512,
    n_mlp=8,
    ckpt='stylegan2-ffhq-config-f.pt'
):
    g_ema = CustomGenerator(size, latent, n_mlp, channel_multiplier=channel_multiplier)
    checkpoint = torch.load(get_path(ckpt))
    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
    g_ema.requires_grad_(False)
    g_ema.eval()
    return g_ema


def bilinear_interpolate_torch(im, y, x):
    """
    im : B,C,H,W
    y : 1,numPoints -- pixel location y float
    x : 1,numPOints -- pixel location y float
    """
    device = im.device

    x0 = torch.floor(x).long().to(device)
    x1 = x0 + 1

    y0 = torch.floor(y).long().to(device)
    y1 = y0 + 1

    wa = ((x1.float() - x) * (y1.float() - y)).to(device)
    wb = ((x1.float() - x) * (y - y0.float())).to(device)
    wc = ((x - x0.float()) * (y1.float() - y)).to(device)
    wd = ((x - x0.float()) * (y - y0.float())).to(device)
    # Instead of clamp
    x1 = x1 - torch.floor(x1 / im.shape[3]).int().to(device)
    y1 = y1 - torch.floor(y1 / im.shape[2]).int().to(device)
    Ia = im[:, :, y0, x0]
    Ib = im[:, :, y1, x0]
    Ic = im[:, :, y0, x1]
    Id = im[:, :, y1, x1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def drag_gan(g_ema, latent: torch.Tensor, noise, F, handle_points, target_points, mask, max_iters=1000):
    handle_points0 = copy.deepcopy(handle_points)
    n = len(handle_points)
    r1, r2, lam, d = 3, 12, 20, 1

    def neighbor(x, y, d):
        points = []
        for i in range(x - d, x + d):
            for j in range(y - d, y + d):
                points.append(torch.tensor([i, j]).float().cuda())
        return points

    F0 = F.detach().clone()

    latent_trainable = latent[:, :6, :].detach().clone().requires_grad_(True)
    latent_untrainable = latent[:, 6:, :].detach().clone().requires_grad_(False)
    optimizer = torch.optim.Adam([latent_trainable], lr=2e-3)
    for iter in range(max_iters):
        for s in range(1):
            optimizer.zero_grad()
            latent = torch.cat([latent_trainable, latent_untrainable], dim=1)
            sample2, F2 = g_ema.generate(latent, noise)

            # motion supervision
            loss = 0
            for i in range(n):
                pi, ti = handle_points[i], target_points[i]
                di = (ti - pi) / torch.sum((ti - pi)**2)

                for qi in neighbor(int(pi[0]), int(pi[1]), r1):
                    # f1 = F[..., int(qi[0]), int(qi[1])]
                    # f2 = F2[..., int(qi[0] + di[0]), int(qi[1] + di[1])]
                    f1 = bilinear_interpolate_torch(F2, qi[0], qi[1]).detach()
                    f2 = bilinear_interpolate_torch(F2, qi[0] + di[0], qi[1] + di[1])
                    loss += FF.l1_loss(f2, f1)

            if mask is not None:
                loss += ((F2 - F0) * (1 - mask)).abs().mean() * lam

            loss.backward()
            optimizer.step()

        # point tracking
        with torch.no_grad():
            sample2, F2 = g_ema.generate(latent, noise)
            for i in range(n):
                pi = handle_points0[i]
                # f = F0[..., int(pi[0]), int(pi[1])]
                f0 = bilinear_interpolate_torch(F0, pi[0], pi[1])
                minv = 1e9
                minx = 1e9
                miny = 1e9
                for qi in neighbor(int(handle_points[i][0]), int(handle_points[i][1]), r2):
                    # f2 = F2[..., int(qi[0]), int(qi[1])]
                    try:
                        f2 = bilinear_interpolate_torch(F2, qi[0], qi[1])
                    except:
                        import ipdb
                        ipdb.set_trace()
                    v = torch.norm(f2 - f0, p=1)
                    if v < minv:
                        minv = v
                        minx = int(qi[0])
                        miny = int(qi[1])
                handle_points[i][0] = minx
                handle_points[i][1] = miny

        F = F2.detach().clone()
        if iter % 1 == 0:
            print(iter, loss.item(), handle_points, target_points)
            # p = handle_points[0].int()
            # sample2[0, :, p[0] - 5:p[0] + 5, p[1] - 5:p[1] + 5] = sample2[0, :, p[0] - 5:p[0] + 5, p[1] - 5:p[1] + 5] * 0
            # t = target_points[0].int()
            # sample2[0, :, t[0] - 5:t[0] + 5, t[1] - 5:t[1] + 5] = sample2[0, :, t[0] - 5:t[0] + 5, t[1] - 5:t[1] + 5] * 255

            # sample2[0, :, 210, 134] = sample2[0, :, 210, 134] * 0
            # utils.save_image(sample2, "test2.png", normalize=True, range=(-1, 1))

        yield sample2, latent, F2, handle_points
