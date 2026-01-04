import os

from torchvision.utils import save_image


def save_generated_images(images, path="generated_images/resultat.png"):

    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    save_image(images, path, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import torch
    import yaml
    from torchvision.utils import save_image

    from diffusion.noise_schedule import Diffusion
    from models.unet import Unet

    checkpoint_path = "models_save/Unet_V0_epoch_100.pt"

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = config["training"]["device"]
    model = Unet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        base_channels=config["model"]["base_channels"],
        time_embedding_dim=config["model"]["time_dim"],
    ).to(device)

    print(f"Chargement de {checkpoint_path}...")
    loaded_content = torch.load(checkpoint_path, map_location=device)

    if isinstance(loaded_content, dict) and "model_state_dict" in loaded_content:
        model.load_state_dict(loaded_content["model_state_dict"])
    else:
        model.load_state_dict(loaded_content)

    diffusion = Diffusion(img_size=config["dataset"]["img_size"], device=device)

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

    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(generated_images.shape[0]):
        filename = f"gen_image_64_{i}.png"
        save_path = os.path.join(output_dir, filename)
        single_image = generated_images[i]
        save_image(single_image, save_path, normalize=True, value_range=(-1, 1))
