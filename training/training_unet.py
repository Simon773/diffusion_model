import os

import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import CelebADataset
from diffusion.noise_schedule import Diffusion
from models.unet import Unet


class Training_Unet:
    def __init__(
        self,
        run_name,
        img_size,
        path_images,
        batch_size,
        lr,
        epochs,
        device,
        time_embedding_dim,
        in_channels,
        out_channels,
    ):
        self.run_name = run_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.path_images = path_images

        self.dataset = CelebADataset(
            path_images=self.path_images, image_size=self.img_size
        )

        self.criterion = nn.MSELoss()

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        self.model = Unet(
            in_channels=in_channels,
            out_channels=out_channels,
            time_embedding_dim=time_embedding_dim,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.diffusion = Diffusion(img_size=self.img_size, device=self.device)

    def train(self):
        print(f"Debut entrainement avec {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            # barre of progress to see the training
            pbar = tqdm(self.dataloader)
            running_loss = 0.0

            for i, images in enumerate(pbar):

                images = images.to(self.device)

                # choose random time steps for each image in the batch
                t = self.diffusion.sample_timesteps(images.shape[0])

                # we add noise to the images according to the time step
                # we obtain the noisy images x_t and the true noise added
                x_t, noise = self.diffusion.noise_images(images, t)

                # prediction of the noise by the model
                predicted_noise = self.model(x_t, t)

                # compute the loss
                loss = self.criterion(predicted_noise, noise)

                # update the model weights
                self.optimizer.zero_grad()  # reset gradients
                loss.backward()  # compute gradients
                self.optimizer.step()  # apply change to weights

                # write the loss on the progress bar
                running_loss += loss.item()
                pbar.set_postfix(MSE=loss.item())

            # end of epoch
            avg_loss = running_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.epochs} | mean loss: {avg_loss:.4f}")

            # save the model every 10 epochs
            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        save_path = os.path.join("models_saved", f"{self.run_name}_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), save_path)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        trainer = Training_Unet(
            run_name=config["run_name"],
            img_size=config["dataset"]["img_size"],
            path_images=config["dataset"]["path"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            epochs=config["training"]["epochs"],
            device=config["training"]["device"],
            time_embedding_dim=config["model"]["time_dim"],
            in_channels=config["model"]["in_channels"],
            out_channels=config["model"]["out_channels"],
        )
        trainer.train()
