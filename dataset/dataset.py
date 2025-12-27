import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CelebADataset(Dataset):
    def __init__(self, path_images, image_size=64):
        self.path_images = path_images
        self.image_size = image_size

        self.image_files = [
            f for f in os.listdir(self.path_images) if f.lower().endswith(".jpg")
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # resize images
                transforms.ToTensor(),  # normalize to [0, 1]
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                # to noralize from [0, 1] to [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.path_images, img_name)
        image = Image.open(img_path).convert("RGB")

        # for resize and normalize images
        image = self.transform(image)
        return image
