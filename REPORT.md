# Diffusion model with U-Net using Celeba Dataset
## Introduction : 
This report details the implementation of a Denoising Diffusion Probabilistic Model (DDPM) applied to the **CelebA** dataset. The goal is to generate high-quality, diverse images of human faces by learning to reverse a gradual noise addition process.

The model for realize this task is U-Net neural network tasked with predicting the noise added to an image at a given timestep $t$. This project explores the mathematical foundations of diffusion, the architectural choices of the U-Net, and the results obtained after training.

## Mathematical Framework

Diffusion models can be represented by a Markov chain. At each step we slowly add random noise (gaussian) to data and then learn to reverse the diffusion process to construct desired data samples from the noise.

### Forward Process 
The forward process is a fixed Markov chain that gradually add gaussian noise to the data (images in RGB) according to a variance schedule ($\beta_1, \dots, \beta_T$). 
Given a data point $x_0$ sampled from the real data distribution, we define the forward transition as:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$
Since it's a markov process the data point $x_t$ depend only on the data points in timestep $t-1$, $x_{t-1}$