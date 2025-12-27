import random

import matplotlib.pyplot as plt

from dataset.dataset import CelebADataset


def show_image(tensor_img):

    # de normalized (from -1, 1) to (0, 1)
    img = (tensor_img + 1) / 2
    img = img.clamp(0, 1)

    # change order of dimensions for malplotlib (H, W, C)
    img = img.permute(1, 2, 0)

    plt.imshow(img)
    plt.axis("off")
    plt.show()


# --- Test ---
if __name__ == "__main__":

    dataset = CelebADataset(path_images="./data/img_align_celeba", image_size=64)

    # take random image
    nb = random.randint(0, len(dataset) - 1)
    image_tensor = dataset[nb]
    show_image(image_tensor)
