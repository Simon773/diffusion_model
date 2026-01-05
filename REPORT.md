# Diffusion model with U-Net using Celeba Dataset
## Introduction : 
This report details the implementation of a Denoising Diffusion Probabilistic Model (DDPM) applied to the **CelebA** dataset. The goal is to generate high-quality, diverse images of human faces by learning to reverse a gradual noise addition process.

The model for realize this task is U-Net neural network tasked with predicting the noise added to an image at a given timestep $t$. This project explores the mathematical foundations of diffusion, the architectural choices of the U-Net, and the results obtained after training.

## Mathematical Framework

Diffusion models can be represented by a Markov chain. At each step we slowly add random noise (gaussian) to data and then learn to reverse the diffusion process to construct desired data samples from the noise.

### Forward Process 
The forward process is a fixed Markov chain that gradually add gaussian noise to the data (images in RGB) according to a variance schedule ($\beta_1, \dots, \beta_T$). 
Given a data point $x_0$ sampled from the real data distribution, we define the forward transition as:
The transition from a image $x_{t-1}$ to a more noisier version  $x_t$ can be expressed with this formula

$$x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon$$

where: $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ represents the gaussian noise sampled at each step.
$\beta_t$ is the variance schedule parameter (controlling the noise magnitude). $\beta_t$ can be fixed for each timestep or we can slowly increase at each iteration.

Since it's a markov process the data point $x_t$ depend only on the data points in timestep $t-1$, $x_{t-1}$


### Reverse Diffusion Process
Instead of dealing with intractable posterior distributions directly, we implement the reverse process as an iterative denoising loop. 

We start from pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, we progressively denoise the image to reach $x_0$.

The transition from step $t$ to $t-1$ is computed using the following update rule, which corresponds directly to our implementation:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

where $x_t$ is the noisy image at the current step.
$\epsilon_\theta(x_t, t)$ is the **noise predicted** by the U-Net (we will see later the role of U-Net).
$\bar{\alpha}_t$ is the cumulative product of alphas.
$z \sim \mathcal{N}(0, \mathbf{I})$ is random noise added to maintain stochasticity (represented by `noise` in the code), except for the final step where $z=0$.
$\sigma_t = \sqrt{\beta_t}$ scales this added noise.

The main idea is at each step, we have an image that is noise. We start with a pure gaussian at timestep $t=1000$
We use the U-Net Neural Network (that is train for this task) to predict the noise of the images. 
Then by the previous formula we remove a part of this predicted noise and add some noise to preserve generative diversity.
At the last step (timestep $t=0), we don't have noise and we have a generated image (normally without noise)