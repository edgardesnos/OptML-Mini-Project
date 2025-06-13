README

# Project Description

This project focuses on a comparative analysis of adversarial attack methods, including both first-order and zeroth-order approaches. A fine-tuned ResNet-18 model is used on the CIFAR-10 dataset to evaluate the effectiveness of different attack techniques. For image generation, a finetuned ResNet50 model is used which can be found in models folder. 

## Code Files

### 1. `optml.ipynb`
This notebook contains all the main code. It includes:
- Loading and preprocessing the CIFAR-10 dataset
- Fine-tuning a pretrained ResNet-18 model for CIFAR-10
- Implementing first-order adversarial attacks (e.g., FGSM, PGD) and zeroth-order attacks(e.g., ZOO with Adam, ZOPGD)
- Comparing the model’s clean accuracy and adversarial accuracy

All experimental steps are included in this file.

### 2. `image_generation.ipynb`
This notebook is used to generate adversarial images based on zeroth-order attacks (ZOO).  
It focuses on producing and visualizing adversarial examples without requiring gradient information.

## Usage

Run `optml.ipynb` to apply first-order and zeroth-order attacks and evaluate their performances.  
Run`zo_image_generation.ipynb` to generate adversarial images using zeroth-order methods.

Both notebooks are designed to run in Jupyter Notebook environments.
