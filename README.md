
# 🧠 Diffusion Models (DDPM) Implementation on CelebA

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![CelebA](https://img.shields.io/badge/Dataset-CelebA-green?style=for-the-badge)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

This project provides an in-depth study and practical implementation of **Denoising Diffusion Probabilistic Models (DDPM)** for high-quality image synthesis. The model is trained on the **CelebA** face dataset using a custom **U-Net** architecture as the noise prediction network.

![Generated Results](generated_images/gen_image_128_1.png)
*(Exemple de résultats générés après entraînement)*

---

## Objectives

The core technical challenge involves designing and integrating the U-Net architecture to master the iterative denoising process.

### 1. Theoretical Foundation
- **Forward Process ($q$):** Implementation of the noise schedule ($\beta_t$) to progressively add Gaussian noise to images.
- **Reverse Process ($p_\theta$):** Learning to reverse the diffusion process to recover $x_0$ from pure noise $x_T$.

### 2. U-Net Architecture
- Adaptation of the classic U-Net (segmentation) for **$\epsilon$-prediction**.
- Implementation of **Sinusoidal Time Embeddings** to inform the network of the current noise level.
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