import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import STL10
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split, Subset
from torch.nn.functional import interpolate
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


#####################################################################################################
# Utils for projected attacks
#####################################################################################################

def normalize(tensor, mode="l_inf"):

    if mode == "l_inf":
        return tensor / tensor.abs().max()

    elif mode == "l_2":
        return tensor / tensor.norm(p=2)




def project_to_ball(tensor, center, radius, mode="l_inf"):

    if mode == "l_inf":
        return torch.clamp(tensor, center - radius, center + radius)

    elif mode == "l_2":
        diff = tensor - center
        norm = diff.norm(p=2)

        if norm > radius:
            diff = diff * (radius / norm)

        return center + diff




def loss_fn(model, x, labels):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    return loss



def estimate_gradient_batched(f, x, h=1e-3, num_dirs=20):
    """
    Estimates the gradient of f at point x using finite differences in random directions.

    Args:
        f: Function that maps [B, C, H, W] tensor to [B] loss values
        x: Tensor of shape [C, H, W]
        h: Finite difference step size
        num_dirs: Number of random directions to use

    Returns:
        grad_estimate: Tensor of shape [C, H, W]
    """

    if x.dim() == 4:
        x = x.squeeze(0)  # convert [1, C, H, W] → [C, H, W]

    device = x.device
    C, H, W = x.shape


    # Generate random directions (normalized Gaussian)
    directions = torch.randn(num_dirs, C, H, W, device=device)
    directions = directions / directions.view(num_dirs, -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-8).view(num_dirs, 1, 1, 1)

    # Create perturbed batches
    x = x.unsqueeze(0)  # [1, C, H, W]
    x_plus = x + h * directions      # [num_dirs, C, H, W]
    x_minus = x - h * directions     # [num_dirs, C, H, W]
    batch = torch.cat([x_plus, x_minus], dim=0)  # [2*num_dirs, C, H, W]

    # Evaluate function in batch
    with torch.no_grad():
        losses = f(batch)  # shape: [2 * num_dirs]

    f_plus = losses[:num_dirs]
    f_minus = losses[num_dirs:]

    # Estimate gradient
    grad_estimate = ((f_plus - f_minus).view(num_dirs, 1, 1, 1) * directions).mean(dim=0) / (2 * h)

    return grad_estimate



#####################################################################################################
# Utils for ZOO attacks
#####################################################################################################


def l2_norm_squared_gradient(a,b):
    return 2 * (a - b)


def linf_norm_subgradient(a,b):
    diff = a - b
    abs_diff = diff.abs()
    max_val = abs_diff.max()
    mask = (abs_diff == max_val)
    subgrad = mask.float() * diff.sign()
    return subgrad



def regularization_gradient(x, y, mode="l_2"):
  if mode == "l_2":
    return l2_norm_squared_gradient(x, y)
  elif mode == "l_inf":
    return linf_norm_subgradient(x, y)
  else:
    raise ValueError("Invalid mode. Must be 'l_2' or 'l_inf'.")


def batched_margin_loss(model, x, labels, kappa=0.0):
    """
    logits: [B, K] - raw logits
    labels: [B] - true or target class indices
    kappa: float - confidence margin (e.g., 0.0 for standard CW loss)

    Returns: loss tensor of shape [B]
    """

    logits = model(x)

    B, K = logits.shape
    device = logits.device

    log_probs = F.log_softmax(logits, dim=1)  # [B, K]

    # Gather log probs of the target labels
    log_true = log_probs[torch.arange(B, device=device), labels]  # [B]

    # Mask out the true class
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    mask.scatter_(1, labels.view(-1, 1), False)

    # Replace true class logits with very low value before max
    log_probs_masked = log_probs.masked_fill(~mask, float('-inf'))

    # Max over other classes
    log_max_other = log_probs_masked.max(dim=1).values  # [B]

    margin = log_true - log_max_other  # [B]
    loss = torch.clamp(margin, min=-kappa)

    return loss  # per-example margin loss




#####################################################################################################
# Utils for Badit Attacks
#####################################################################################################

def estimate_gradient_bandit_batched(f, x, v, h1, h2, num_dirs=20):
    """
    Estimates the gradient of f at point x using finite differences in random directions.
    """

    if x.dim() == 4:
        x = x.squeeze(0)  # convert [1, C, H, W] → [C, H, W]

    device = x.device
    C, H, W = x.shape


    # Generate random directions (normalized Gaussian)
    directions = torch.randn(num_dirs, C, H, W, device=device)
    directions = directions / directions.view(num_dirs, -1).norm(p=2, dim=1, keepdim=True).clamp(min=1e-8).view(num_dirs, 1, 1, 1)

    # Create perturbed batches
    x = x.unsqueeze(0)  # [1, C, H, W]
    q1 = v + h1 * directions      # [num_dirs, C, H, W]
    q2 = v - h1 * directions     # [num_dirs, C, H, W]

    x_plus = x + h2 * q2
    x_minus = x + h2 * q1
    batch = torch.cat([x_plus, x_minus], dim=0)  # [2*num_dirs, C, H, W]

    # Evaluate function in batch
    with torch.no_grad():
        losses = f(batch)  # shape: [2 * num_dirs]

    f_plus = losses[:num_dirs]
    f_minus = losses[num_dirs:]

    # Estimate gradient
    grad_estimate = ((f_plus - f_minus).view(num_dirs, 1, 1, 1) * directions).mean(dim=0) / (h1 * h2)

    return grad_estimate




#####################################################################################################
# Utils for Square Attack
#####################################################################################################

def square_loss(model, x, label):
    """
    Compute the margin loss L(f(x), y) = f_y(x) - max_{k ≠ y} f_k(x)
    for a single input image and label.

    """
    if x.dim() == 3:
        x = x.unsqueeze(0)  # [1, C, H, W]

    x_resized = interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

    logits = model(x_resized)  # [1, K]
    logits = logits.squeeze(0)  # [K]

    correct_logit = logits[label]

    logits_masked = logits.clone()
    logits_masked[label] = float('-inf')

    max_wrong_logit = logits_masked.max().item()


    margin = correct_logit.item() - max_wrong_logit

    return margin  # float


def sample_direction(epsilon, w, c, window_size):

    delta = torch.zeros((c, w, w))
    delta = torch.zeros((1, c, w, w))


    r = random.randint(0, w - window_size)
    s = random.randint(0, w - window_size)


    for i in range(c):
        rho = random.choice([-2 * epsilon, 2 * epsilon])
        delta[0, i, r:r + window_size, s:s + window_size] = rho

    return delta


def h_schedule(step, N, image_size=96):
    """
    Returns square size h for step in Square Attack
    """
    base_sizes = [16, 14, 12, 10, 8, 6, 4]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

    for threshold, h in zip(thresholds, base_sizes):
        if step < threshold * N:
            return h



#####################################################################################################
# Utils for Visualization
#####################################################################################################




# Unnormalize function to reverse the transform.Normalize
def unnormalize(img_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(3, 1, 1)
    return img_tensor * std + mean

# Function to plot side-by-side images
def plot_original_vs_adversarial(original, adversarial, true_label, adv_label, title=None):

    stl10_labels = {
    0: 'airplane',
    1: 'bird',
    2: 'car',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'horse',
    7: 'monkey',
    8: 'ship',
    9: 'truck'
    }

    # Detach and move to CPU
    original_img = unnormalize(original.squeeze().detach()).cpu().permute(1, 2, 0).numpy()
    adversarial_img = unnormalize(adversarial.squeeze().detach()).cpu().permute(1, 2, 0).numpy()

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(original_img)
    axs[0].set_title(f"Original Image \nLabel: {stl10_labels[true_label]}")
    axs[0].axis("off")

    axs[1].imshow(adversarial_img)
    axs[1].set_title(f"Adversarial Image \nLabel: {stl10_labels[adv_label]}")
    axs[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()




