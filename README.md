
# 🧠 Diffusion Models (DDPM) Implementation on CelebA

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

This project show the implementation of Denoising Diffusion Probabilistic Models (DDPM) for high-quality image synthesis. The model is trained on the CelebA face dataset using a custom **U-Net** architecture as the noise prediction network.

![Generated Results](generated_images/gen_image_128_1.png)



## Objectives

The core technical challenge involves designing and integrating the U-Net architecture to master the iterative denoising process.

### 1. Theoretical Foundation
- **Forward Process :** Implementation of the noise schedule ($\beta_t$,$\alpha_t$) to progressively add Gaussian noise to images.
- **Reverse Process :** Learning to reverse the diffusion process to recover $x_0$ from pure noise $x_T$.

### 2. U-Net Architecture
- Adaptation of the classic U-Net (segmentation) for **$\epsilon$-prediction**.
- Implementation of Sinusoidal Time Embeddings to inform the network of the current noise level.
- Use of **Skip Connections** to preserve fine spatial details.

### 3. Data Pipeline
- Robust handling of the **CelebA dataset**.
- Preprocessing: Resizing (64x64 or 128x128) and Normalization to the range $[-1, 1]$.

---

##  Installation

To run this project you need to create a conda environnement.


1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Simon773/diffusion_model.git](https://github.com/Simon773/diffusion_model.git)
   cd diffusion_model
   ```
Then for create the conda environnement and download required packages, you can use this command :
   ```bash
    conda create --name diffusion_env python=3.10 -y
    conda activate diffusion_env

    pip install -r requirements.txt
   ```

## Project Structure : 

- **config.yaml** : A file defining hyperparameters for the training and the generation of images. You can adjust the size of images, batch, epochs...

- **data** : This folder is not provided in this repo but it contains all the data necessary for the training. You can find data here : [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

- **dataset** : This file allow to transform images for training (resizing,normalizing,transformation to a tensor...)

- **diffusion** : This folder contain only one file : noise_schedule.py. There are functions for add noise to an image, reverse the noise.

