import torch
from tqdm import tqdm


class Diffusion:
    def __init__(
        self,
        img_size,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
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

    def reverse_diffusion(self, model, n_samples):
        """
        Generates samples by reversing the diffusion process
        """
        model.eval()

        with torch.no_grad():
            # we start from pure noise
            x = torch.randn((n_samples, 3, self.img_size, self.img_size)).to(
                self.device
            )

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # we create a tensor filled with the timestep i of size n_samples
                t = (torch.ones(n_samples) * i).long().to(self.device)
                # the model predicts the noise at step t
                predicted_noise = model(x, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                # we add noise only if we are not in the last step
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # equation of sampling to get x t-1 from x_t
                x = (1 / torch.sqrt(alpha)) * (
                    x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
                ) + torch.sqrt(beta) * noise

        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)

        return x
