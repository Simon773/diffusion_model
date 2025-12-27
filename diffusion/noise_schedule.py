import torch


class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=64,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.linspace_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def linspace_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Applies noise to the original images 'x' at timestep 't'.

        Input:
            x  tensor of original images (Batch, Channels, Height, Width)
            t batch of time step (ex : tensort([20,50,10]))

        Output:
            x_t : noisy images
            noise: The actual noise added (the target for the network)
        """
        # generate the noise with a gaussian N(0,1)
        epsilon = torch.randn_like(x)

        # take coefficient of alpha_hat a time t and convert to the good format
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[
            :, None, None, None
        ]

        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon

        return x_t, epsilon

    def sample_timesteps(self, n):
        # generates random timesteps for training of size n
        return torch.randint(
            low=1, high=self.noise_steps, size=(n,), device=self.device
        )
