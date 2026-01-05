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


### Architecture of U-Net

The noise predictor $\epsilon_\theta(x_t, t)$ is implemented as a U-Net, an architecture originally designed for biomedical image segmentation. (article : U-Net: Convolutional Networks for Biomedical Image Segmentation by O. Ronneberger et al)


The U-Net consists of a contracting path (Encoder) and an expansive path (Decoder), giving it its "U" shape.
- Encoder (Downsampling) Progressively reduces spatial resolution while double the number of feature maps (or channel depth). We use a block with MaxPooling followed by 2 Convolutions.
- Bottleneck: The deepest part of the network where the representation is most abstract. In the botteleneck,the size of feature map very low and sometimes we can add a self attention block to increase precision.   
- Decoder (Upsampling): Restores spatial resolution using or Upsampling layers. In each Upsampling layers, we divided by 2 the number of feature maps
- Skip Connections: Concatenate the feature maps from the Encoder to the corresponding layers in the Decoder. This is crucial for diffusion models as it allows the network to retain  details lost during downsampling.

#### Time Embeddings
Since the U-Net shares parameters across all timesteps $t$, it must be conditioned on $t$ to know the noise level.
We do not simply feed the scalar $t$ (which would be too weak a signal). Instead, we project $t$ into a high-dimensional vector space using Sinusoidal Positional Embeddings, a technique adapted from the Transformer architecture.

or a given timestep $t$ and an embedding dimension $d$ (e.g., 64 or 32), the embedding vector $e_t$ is constructed using pairs of sine and cosine functions with geometrically progressing frequencies:

$$
\begin{aligned}
e_t[2i] &= \sin(t / 10000^{2i/d}) \\
e_t[2i+1] &= \cos(t / 10000^{2i/d})
\end{aligned}
$$
where $i$ represents the dimension index from $0$ to $d/2$.
Once we have the raw vector, we create a MLP: The fixed sinusoidal vector passes through a small Multi-Layer Perceptron (Linear $\to$ SiLU (smooth ReLU) Activation $\to$ Linear). This allows the network to learn a non-linear representation of time specifically for the denoising task.

Then once we have this vector, we add it in the U-Net, at each depth of the Unet between the 2 convolutions. This allows to keep information of time in each layer of the network. 

### Implementation Details
* **Input/Output:** 3 channels (RGB) $\to$ 3 channels (Predicted Noise).
* **Dimensions:** 64 base channels (can be change to 128 or more)
* **Activation:** ReLu but can be change to SiLU or Switch(?)that are better for gradient.
* **Normalization:** BatchNorm is use but some articles states stat GroupNorm is preferred over BatchNorm for smaller batch sizes typical in generative tasks.

The model was trained for 100 epochs using the Adam optimizer. In the first model, I use a size of 64 pixels for image size and 128 in the second model. The model can be extend with images with more pixels but since the time of training is exponential of the image's size, this project is limited to small resolution images. (around 42h in Kaggle Notebook to train U-net model in 128*128 )