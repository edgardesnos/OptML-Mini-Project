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
from zo_utils import*



#####################################################################################################
# Projected Attacks
#####################################################################################################

def zero_pgd_attack(
    model,
    original_image,
    true_label,
    steps=30,
    lr=0.01,
    delta=1e-3,
    epsilon=1e-1,
    num_dirs=20,
    targeted=False
):



    def loss_wrapper(z_batch):
        """
        z_batch: Tensor of shape [B, C, H, W] — perturbed images
        true_label: integer class index (from outer scope)
        """
        if z_batch.dim() == 3:
            z_batch = z_batch.unsqueeze(0)  # add batch dim if single image

        # Repeat label for each sample in batch
        labels = torch.full((z_batch.size(0),), true_label, dtype=torch.long, device=device)

        # Resize if needed (e.g., to match ResNet input)
        z_batch_resized = torch.nn.functional.interpolate(
            z_batch, size=(224, 224), mode='bilinear', align_corners=False
        )

        #logits = model(z_batch_resized)
        #losses = torch.nn.functional.cross_entropy(logits, labels, reduction='none')  # shape: [B]

        return loss_fn(model, z_batch_resized, labels)  # return per-sample loss values


    with torch.no_grad():
        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach().to(device)
        x_adv.requires_grad = False

        for step in range(steps):
            grad = estimate_gradient_batched(loss_wrapper, x_adv, h=delta, num_dirs=num_dirs)

            grad = normalize(grad, mode="l_inf")


            x_adv = x_adv + lr * grad

            x_adv = project_to_ball(x_adv, x, radius=epsilon, mode="l_inf")

            x_adv = torch.clamp(x_adv, -1, 1)

            # Optional: Print confidence

            x_resized = interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 10 == 0:
              print(f"Step {step+1}/{steps}: Prediction={pred}, Confidence={confidence:.4f}")

            if pred != true_label:
                print("Adversarial example found!")
                break

        return x_adv, pred



def zero_pgd_attack_adam(
    model,
    original_image,
    true_label,
    steps=30,
    lr=0.01,
    delta=1e-3,
    epsilon=1e-1,
    num_dirs=20,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    targeted=False
):
    def loss_wrapper(z_batch):
        if z_batch.dim() == 3:
            z_batch = z_batch.unsqueeze(0)
        labels = torch.full((z_batch.size(0),), true_label, dtype=torch.long, device=device)
        z_batch_resized = torch.nn.functional.interpolate(
            z_batch, size=(224, 224), mode='bilinear', align_corners=False
        )
        return loss_fn(model, z_batch_resized, labels)

    with torch.no_grad():
        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach().to(device)

        m = torch.zeros_like(x_adv)
        v = torch.zeros_like(x_adv)

        for step in range(1, steps + 1):
            grad = estimate_gradient_batched(loss_wrapper, x_adv, h=delta, num_dirs=num_dirs)
            grad = normalize(grad, mode="l_inf")

            # If targeted, reverse the direction
            if targeted:
                grad = -grad

            # Adam updates
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad.pow(2)

            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            step_update = lr * m_hat / (v_hat.sqrt() + eps)
            x_adv = x_adv + step_update

            x_adv = project_to_ball(x_adv, x, radius=epsilon, mode="l_inf")
            x_adv = torch.clamp(x_adv, -1, 1)

            # Print progress
            x_resized = torch.nn.functional.interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 10 == 0 or pred != true_label:
                print(f"Step {step}/{steps}: Prediction={pred}, Confidence={confidence:.4f}")

            if pred != true_label:
                print("Adversarial example found!")
                break

        return x_adv, pred



#####################################################################################################
# ZOO Attacks
#####################################################################################################

def zoo_attack(
    model,
    original_image,
    true_label,
    steps=30,
    alpha=1.0,
    delta=1e-3,
    num_dirs=20,
    lr=0.1,
    targeted=False
):
    """
    ZOO-style black-box gradient descent adversarial attack on a single image.
    - original_image: [1, 3, 96, 96] (torch.Tensor)
    - true_label: int
    """



    def loss_wrapper(z_batch):
        """
        z_batch: Tensor of shape [B, C, H, W] — perturbed images
        true_label: integer class index (from outer scope)
        """
        if z_batch.dim() == 3:
            z_batch = z_batch.unsqueeze(0)  # add batch dim if single image

        # Repeat label for each sample in batch
        labels = torch.full((z_batch.size(0),), true_label, dtype=torch.long, device=device)

        # Resize if needed (e.g., to match ResNet input)
        z_batch_resized = torch.nn.functional.interpolate(
            z_batch, size=(224, 224), mode='bilinear', align_corners=False
        )

        return batched_margin_loss(model, z_batch_resized, labels)  # return per-sample loss values


    with torch.no_grad():
        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach().to(device)
        x_adv.requires_grad = False

        for step in range(steps):
            grad_loss = estimate_gradient_batched(loss_wrapper, x_adv, h=delta, num_dirs=num_dirs)

            grad_reg = regularization_gradient(x_adv, x, mode="l_2")



            grad = grad_loss + alpha * grad_reg

            x_adv = x_adv - lr*grad

            x_adv = torch.clamp(x_adv, -1, 1)

            # Optional: Print confidence

            x_resized = interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 10 == 0:
              print(f"Step {step+1}/{steps}: Prediction={pred}, Confidence={confidence:.4f}")

            if pred != true_label:
                print("Adversarial example found!")
                break

        return x_adv, pred



# ZOO with ADAM
def zoo_attack_adam(
    model,
    original_image,
    true_label,
    steps=500,
    alpha=1.0,
    delta=1e-3,
    num_dirs=100,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    targeted=False
):


    def loss_wrapper(z_batch):
        if z_batch.dim() == 3:
            z_batch = z_batch.unsqueeze(0)
        labels = torch.full((z_batch.size(0),), true_label, dtype=torch.long, device=device)
        z_batch_resized = F.interpolate(z_batch, size=(224, 224), mode='bilinear', align_corners=False)
        return batched_margin_loss(model, z_batch_resized, labels)

    def regularization_gradient(x, y, mode="l_2"):
        if mode == "l_2":
            return 2 * (x - y)
        elif mode == "l_inf":
            diff = x - y
            abs_diff = diff.abs()
            max_val = abs_diff.max()
            mask = (abs_diff == max_val)
            return mask.float() * diff.sign()
        else:
            raise ValueError("Invalid mode. Must be 'l_2' or 'l_inf'.")

    with torch.no_grad():
        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach()
        m = torch.zeros_like(x_adv)
        v = torch.zeros_like(x_adv)

        for step in range(1, steps + 1):
            grad_loss = estimate_gradient_batched(loss_wrapper, x_adv, h=delta, num_dirs=num_dirs)
            grad_reg = regularization_gradient(x_adv, x, mode="l_2")
            grad = grad_loss + alpha * grad_reg

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)

            x_adv = x_adv - lr * m_hat / (v_hat.sqrt() + eps)
            x_adv = torch.clamp(x_adv, -1, 1)

            # Monitor progress
            x_resized = F.interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 10 == 0:
                print(f"Step {step}/{steps}: Prediction={pred}, Confidence={confidence:.4f}")

            if pred != true_label:
                print("Adversarial example found!")
                break

        return x_adv, pred



#####################################################################################################
# Bandit Attacks
#####################################################################################################

def bandit_l2_attack(
    model,
    original_image,
    true_label,
    steps=500,
    lr=0.5,
    eta=0.1,
    delta=1e-2,
    epsilon=1e-2,
    num_dirs=100,
    targeted=False
):
    """
    Bandit-style black-box gradient descent adversarial attack on a single image.
    - original_image: [1, 3, 96, 96] (torch.Tensor)
    - true_label: int
    """


    def loss_wrapper(z_batch):
        """
        z_batch: Tensor of shape [B, C, H, W] — perturbed images
        true_label: integer class index (from outer scope)
        """
        if z_batch.dim() == 3:
            z_batch = z_batch.unsqueeze(0)  # add batch dim if single image

        # Repeat label for each sample in batch
        labels = torch.full((z_batch.size(0),), true_label, dtype=torch.long, device=device)

        # Resize if needed (e.g., to match ResNet input)
        z_batch_resized = torch.nn.functional.interpolate(
            z_batch, size=(224, 224), mode='bilinear', align_corners=False
        )

        #logits = model(z_batch_resized)
        #losses = torch.nn.functional.cross_entropy(logits, labels, reduction='none')  # shape: [B]

        return loss_fn(model, z_batch_resized, labels)  # return per-sample loss values


    with torch.no_grad():

        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach().to(device)
        x_adv.requires_grad = False

        v = torch.zeros_like(x)  # latent direction (same shape as image)

        for step in range(steps):
            grad = estimate_gradient_bandit_batched(loss_wrapper, x_adv, v, h1=delta, h2=epsilon, num_dirs=num_dirs)

            v = v + eta * grad


            g = normalize(v, mode="l_2")


            x_adv = x_adv + lr * g

            x_adv = project_to_ball(x_adv, x, radius=epsilon, mode="l_2")

            x_adv = torch.clamp(x_adv, -1, 1)

            # Optional: Print confidence

            x_resized = interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 10 == 0:
              print(f"Step {step+1}/{steps}: Prediction={pred}, Confidence={confidence:.4f}")

            if pred != true_label:
                print("Adversarial example found!")
                break

        return x_adv, pred



def bandit_linf_attack(
    model,
    original_image,
    true_label,
    steps=500,
    lr=0.5,
    eta=0.1,
    delta=1e-2,
    epsilon=0.05,
    num_dirs=100,
    targeted=False
):
    def loss_wrapper(z):
        y = torch.tensor([true_label], device=device)
        return loss_fn(model, z, y)

    def loss_wrapper_2(z_batch):
        if z_batch.dim() == 3:
            z_batch = z_batch.unsqueeze(0)
        labels = torch.full((z_batch.size(0),), true_label, dtype=torch.long, device=device)
        z_batch_resized = torch.nn.functional.interpolate(
            z_batch, size=(224, 224), mode='bilinear', align_corners=False
        )
        return loss_fn(model, z_batch_resized, labels)

    with torch.no_grad():
        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach().to(device)
        x_adv.requires_grad = False

        v = torch.zeros_like(x)

        for step in range(steps):
            grad = estimate_gradient_bandit_batched(loss_wrapper_2, x_adv, v, h1=delta, h2=epsilon, num_dirs=num_dirs)

            # EG update
            p = (v + 1.0) / 2.0  # Map to [0, 1]
            exp_up = torch.exp(eta * grad)
            exp_down = torch.exp(-eta * grad)
            Z = p * exp_up + (1 - p) * exp_down + 1e-8
            p = p * exp_up / Z
            v = 2 * p - 1  # Map back to [-1, 1]

            g = v  # direction proposal from EG update
            x_adv = x_adv + lr * g
            x_adv = project_to_ball(x_adv, x, radius=epsilon, mode="l_inf")
            x_adv = torch.clamp(x_adv, -1, 1)

            # Optional: Print confidence
            x_resized = interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 10 == 0:
                print(f"Step {step+1}/{steps}: Prediction={pred}, Confidence={confidence:.4f}")

            if pred != true_label:
                print("Adversarial example found!")
                break

        return x_adv, pred



#####################################################################################################
# Square Attack
#####################################################################################################

def square_attack(
    model,
    original_image,
    true_label,
    steps=2000,
    epsilon=0.05,
    targeted=False
):



    with torch.no_grad():
        device = original_image.device
        x = original_image.clone().detach().to(device)
        x_adv = x.clone().detach().to(device)
        x_adv.requires_grad = False
        l = square_loss(model, x, torch.tensor([true_label], device=device))

        [B, C, H, W] = x.shape

        for step in range(steps):

            h = h_schedule(step, steps)

            delta = sample_direction(epsilon, W, C, h).to(device)

            x_new = x_adv + delta

            x_new = project_to_ball(x_new, x, radius=epsilon, mode="l_inf")

            x_new = torch.clamp(x_new, -1, 1)

            l_new = square_loss(model, x_new, torch.tensor([true_label], device=device))

            if l_new < l:
                x_adv = x_new
                l = l_new


            # Print confidence

            x_resized = interpolate(x_adv, size=(224, 224), mode='bilinear', align_corners=False)
            output = model(x_resized)
            pred = output.argmax(dim=1).item()
            confidence = output.softmax(dim=1)[0, pred].item()

            if step % 100 == 0:
              print(f"\nStep {step+1}/{steps}: Prediction={pred}, Confidence={confidence:.4f}, Loss={l:.4f}")


            if pred != true_label:
                print("Adversarial example found!")
                print(f"\nStep {step+1}/{steps}: Prediction={pred}")

                break

        return x_adv, pred
