import os

from torchvision.utils import save_image


def save_generated_images(images, path="/generated_images/resultat.png"):

    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    save_image(images, path, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from torchvision.utils import save_image

    from diffusion.noise_schedule import Diffusion
    from models.unet import Unet

    checkpoint_path = "models_save/Unet_V0_epoch_100.pt"
    device = "gpu"

    model = Unet(
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_embedding_dim=32,
    ).to(device)

    print(f"Chargement de {checkpoint_path}...")
    loaded_content = torch.load(checkpoint_path, map_location=device)

    if isinstance(loaded_content, dict) and "model_state_dict" in loaded_content:
        model.load_state_dict(loaded_content["model_state_dict"])
    else:
        model.load_state_dict(loaded_content)

    diffusion = Diffusion(img_size=128, device="cuda")

    generated_images = diffusion.reverse_diffusion(model, n_samples=1)

    def plot_results(images):
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)

            img = (images[i] + 1) / 2
            img = img.clamp(0, 1)

            img = img.permute(1, 2, 0).cpu().numpy()

            plt.imshow(img)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    plot_results(generated_images)

    save_path = "/generated_images/resultat.png"
    save_image(generated_images, save_path, normalize=True, value_range=(-1, 1))
    print(f"✅ Image sauvegardée sous : {save_path}")
