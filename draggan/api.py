import copy
import random

import torch
import torch.nn.functional as FF
import torch.optim

from . import utils
from .stylegan2.model import Generator


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
    g_ema = CustomGenerator(size, latent, n_mlp, channel_multiplier=channel_multiplier, human='human' in ckpt)
    checkpoint = torch.load(utils.get_path(ckpt))
    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
    g_ema.requires_grad_(False)
    g_ema.eval()
    return g_ema


def drag_gan(
    g_ema,
    latent: torch.Tensor,
    noise,
    F,
    handle_points,
    target_points,
    mask,
    max_iters=1000,
    r1=3,
    r2=12,
    lam=20,
    d=2,
    lr=2e-3,
):
    handle_points0 = copy.deepcopy(handle_points)
    handle_points = torch.stack(handle_points)
    handle_points0 = torch.stack(handle_points0)
    target_points = torch.stack(target_points)

    F0 = F.detach().clone()
    device = latent.device

    latent_trainable = latent[:, :6, :].detach().clone().requires_grad_(True)
    latent_untrainable = latent[:, 6:, :].detach().clone().requires_grad_(False)
    optimizer = torch.optim.Adam([latent_trainable], lr=lr)
    for _ in range(max_iters):
        if torch.allclose(handle_points, target_points, atol=d):
            break

        optimizer.zero_grad()
        latent = torch.cat([latent_trainable, latent_untrainable], dim=1)
        sample2, F2 = g_ema.generate(latent, noise)

        # motion supervision
        loss = motion_supervison(handle_points, target_points, F2, r1, device)

        if mask is not None:
            loss += ((F2 - F0) * (1 - mask)).abs().mean() * lam

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            latent = torch.cat([latent_trainable, latent_untrainable], dim=1)
            sample2, F2 = g_ema.generate(latent, noise)
            handle_points = point_tracking(F2, F0, handle_points, handle_points0, r2, device)

        F = F2.detach().clone()
        # if iter % 1 == 0:
        #     print(iter, loss.item(), handle_points, target_points)

        yield sample2, latent, F2, handle_points


def motion_supervison(handle_points, target_points, F2, r1, device):
    loss = 0
    n = len(handle_points)
    for i in range(n):
        target2handle = target_points[i] - handle_points[i]
        d_i = target2handle / (torch.norm(target2handle) + 1e-7)
        if torch.norm(d_i) > torch.norm(target2handle):
            d_i = target2handle

        mask = utils.create_circular_mask(
            F2.shape[2], F2.shape[3], center=handle_points[i].tolist(), radius=r1
        ).to(device)

        coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None]

        h, w = F2.shape[2], F2.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = F2[:, :, mask]  # shape: [C, H*W]

        # Sample shifted patch from F
        normalized_shifted_coordinates = shifted_coordinates.clone()
        normalized_shifted_coordinates[:, 0] = (
            2.0 * shifted_coordinates[:, 0] / (h - 1)
        ) - 1  # for height
        normalized_shifted_coordinates[:, 1] = (
            2.0 * shifted_coordinates[:, 1] / (w - 1)
        ) - 1  # for width
        # Add extra dimensions for batch and channels (required by grid_sample)
        normalized_shifted_coordinates = normalized_shifted_coordinates.unsqueeze(
            0
        ).unsqueeze(
            0
        )  # shape [1, 1, num_points, 2]
        normalized_shifted_coordinates = normalized_shifted_coordinates.flip(
            -1
        )  # grid_sample expects [x, y] instead of [y, x]
        normalized_shifted_coordinates = normalized_shifted_coordinates.clamp(-1, 1)

        # Use grid_sample to interpolate the feature map F at the shifted patch coordinates
        F_qi_plus_di = torch.nn.functional.grid_sample(
            F2, normalized_shifted_coordinates, mode="bilinear", align_corners=True
        )
        # Output has shape [1, C, 1, num_points] so squeeze it
        F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

        loss += torch.nn.functional.l1_loss(F_qi.detach(), F_qi_plus_di)
    return loss


def point_tracking(
    F: torch.Tensor,
    F0: torch.Tensor,
    handle_points: torch.Tensor,
    handle_points0: torch.Tensor,
    r2: int = 3,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:

    n = handle_points.shape[0]  # Number of handle points
    new_handle_points = torch.zeros_like(handle_points)

    for i in range(n):
        # Compute the patch around the handle point
        patch = utils.create_square_mask(
            F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r2
        ).to(device)

        # Find indices where the patch is True
        patch_coordinates = torch.nonzero(patch)  # shape [num_points, 2]

        # Extract features in the patch
        F_qi = F[:, :, patch_coordinates[:, 0], patch_coordinates[:, 1]]
        # Extract feature of the initial handle point
        f_i = F0[:, :, handle_points0[i][0].long(), handle_points0[i][1].long()]

        # Compute the L1 distance between the patch features and the initial handle point feature
        distances = torch.norm(F_qi - f_i[:, :, None], p=1, dim=1)

        # Find the new handle point as the one with minimum distance
        min_index = torch.argmin(distances)
        new_handle_points[i] = patch_coordinates[min_index]

    return new_handle_points
