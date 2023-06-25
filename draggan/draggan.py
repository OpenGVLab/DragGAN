# Modified from this following version https://github.com/skimai/DragGAN

import os
import sys
import time
from typing import List, Optional, Tuple
import copy

import numpy as np
import PIL
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
stylegan2_dir = os.path.join(CURRENT_DIR, "stylegan2")
sys.path.insert(0, stylegan2_dir)
import dnnlib
import legacy
from . import utils

def load_model(
    network_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
    device: torch.device = torch.device("cuda"),
    fp16: bool = True,
) -> torch.nn.Module:
    """
    Loads a pretrained StyleGAN2-ADA generator network from a pickle file.

    Args:
        network_pkl (str): The URL or local path to the network pickle file.
        device (torch.device): The device to use for the computation.
        fp16 (bool): Whether to use half-precision floating point format for the network weights.

    Returns:
        The pretrained generator network.
    """
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        chkpt = legacy.load_network_pkl(f, force_fp16=fp16)
    G = chkpt["G_ema"].to(device).eval()
    for param in G.parameters():
        param.requires_grad_(False)

    # Create a new attribute called "activations" for the Generator class
    # This will be a list of activations from each layer
    G.__setattr__("activations", None)

    # Forward hook to collect features
    def hook(module, input, output):
        G.activations = output

    # Apply the hook to the 7th layer (256x256)
    for i, (name, module) in enumerate(G.synthesis.named_children()):
        if i == 6:
            print("Registering hook for:", name)
            module.register_forward_hook(hook)

    return G


def register_hook(G):
    # Create a new attribute called "activations" for the Generator class
    # This will be a list of activations from each layer
    G.__setattr__("activations", None)

    # Forward hook to collect features
    def hook(module, input, output):
        G.activations = output

    # Apply the hook to the 7th layer (256x256)
    for i, (name, module) in enumerate(G.synthesis.named_children()):
        if i == 6:
            print("Registering hook for:", name)
            module.register_forward_hook(hook)
    return G


def generate_W(
    _G: torch.nn.Module,
    seed: int = 0,
    network_pkl: Optional[str] = None,
    truncation_psi: float = 1.0,
    truncation_cutoff: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """
    Generates a latent code tensor in W+ space from a pretrained StyleGAN2-ADA generator network.

    Args:
        _G (torch.nn.Module): The generator network, with underscore to avoid streamlit cache error
        seed (int): The random seed to use for generating the latent code.
        network_pkl (Optional[str]): The path to the network pickle file. If None, the default network will be used.
        truncation_psi (float): The truncation psi value to use for the mapping network.
        truncation_cutoff (Optional[int]): The number of layers to use for the truncation trick. If None, all layers will be used.
        device (torch.device): The device to use for the computation.

    Returns:
        The W+ latent as a numpy array of shape [1, num_layers, 512].
    """
    G = _G
    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim).to(device)
    num_layers = G.synthesis.num_ws
    if truncation_cutoff == -1:
        truncation_cutoff = None
    elif truncation_cutoff is not None:
        truncation_cutoff = min(num_layers, truncation_cutoff)
    W = G.mapping(
        z,
        None,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
    )
    return W.cpu().numpy()


def forward_G(
    G: torch.nn.Module,
    W: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through the generator network.

    Args:
        G (torch.nn.Module): The generator network.
        W (torch.Tensor): The latent code tensor of shape [batch_size, latent_dim, 512].
        device (torch.device): The device to use for the computation.

    Returns:
        A tuple containing the generated image tensor of shape [batch_size, 3, height, width]
        and the feature maps tensor of shape [batch_size, num_channels, height, width].
    """
    register_hook(G)
    
    if not isinstance(W, torch.Tensor):
        W = torch.from_numpy(W).to(device)

    img = G.synthesis(W, noise_mode="const", force_fp32=True)

    return img, G.activations[0]


def generate_image(
    W,
    _G: Optional[torch.nn.Module] = None,
    network_pkl: Optional[str] = None,
    class_idx=None,
    device=torch.device("cuda"),
) -> Tuple[PIL.Image.Image, torch.Tensor]:
    """
    Generates an image using a pretrained generator network.

    Args:
        W (torch.Tensor): A tensor of latent codes of shape [batch_size, latent_dim, 512].
        _G (Optional[torch.nn.Module]): The generator network. If None, the network will be loaded from `network_pkl`.
        network_pkl (Optional[str]): The path to the network pickle file. If None, the default network will be used.
        class_idx (Optional[int]): The class index to use for conditional generation. If None, unconditional generation will be used.
        device (str): The device to use for the computation.

    Returns:
        A tuple containing the generated image as a PIL Image object and the feature maps tensor of shape [batch_size, num_channels, height, width].
    """
    if _G is None:
        assert network_pkl is not None
        _G = load_model(network_pkl, device)
    G = _G

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise Exception(
                "Must specify class label with --class when using a conditional network"
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    # Generate image
    img, features = forward_G(G, W, device)

    img = utils.tensor_to_PIL(img)

    return img, features


def drag_gan(
    W,
    G,
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

    device = torch.device("cuda")

    img, F0 = forward_G(G, W, device)

    target_resolution = img.shape[-1]
    F0_resized = torch.nn.functional.interpolate(
        F0,
        size=(target_resolution, target_resolution),
        mode="bilinear",
        align_corners=True,
    ).detach()

    W = torch.from_numpy(W).to(device).float()
    W.requires_grad_(False)

    # Only optimize the first 6 layers of W
    W_layers_to_optimize = W[:, :6].clone()
    W_layers_to_optimize.requires_grad_(True)

    optimizer = torch.optim.Adam([W_layers_to_optimize], lr=lr)

    for _ in range(max_iters):
        start = time.perf_counter()
        if torch.allclose(handle_points, target_points, atol=d):
            break

        optimizer.zero_grad()
        W_combined = torch.cat([W_layers_to_optimize, W[:, 6:].detach()], dim=1)

        img, F = forward_G(G, W_combined, device)
        F_resized = torch.nn.functional.interpolate(
            F,
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=True,
        )

        # motion supervision
        loss = motion_supervison(handle_points, target_points, F_resized, r1, device)

        # if mask is not None:
        #     loss += ((F - F0) * (1 - mask)).abs().mean() * lam

        loss.backward()
        optimizer.step()

        print(
            f"Loss: {loss.item():0.2f}\tTime: {(time.perf_counter() - start) * 1000:.0f}ms"
        )
        
        with torch.no_grad():
            img, F = forward_G(G, W_combined, device)
            handle_points = point_tracking(F_resized, F0_resized, handle_points, handle_points0, r2, device)

        # if iter % 1 == 0:
        #     print(iter, loss.item(), handle_points, target_points)
        W_out = torch.cat([W_layers_to_optimize, W[:, 6:]], dim=1).detach().cpu().numpy()

        img = utils.tensor_to_PIL(img)
        yield img, W_out, handle_points


def motion_supervison(handle_points, target_points, F, r1, device):
    loss = 0
    n = len(handle_points)
    for i in range(n):
        target2handle = target_points[i] - handle_points[i]
        d_i = target2handle / (torch.norm(target2handle) + 1e-7)
        if torch.norm(d_i) > torch.norm(target2handle):
            d_i = target2handle

        mask = utils.create_circular_mask(
            F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r1
        ).to(device)

        coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None]

        h, w = F.shape[2], F.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = F[:, :, mask]  # shape: [C, H*W]

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
            F, normalized_shifted_coordinates, mode="bilinear", align_corners=True
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
