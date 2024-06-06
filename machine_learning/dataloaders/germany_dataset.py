import torch
import os
from torch.utils.data import Dataset
import torchvision.transforms.v2.functional as F
from PIL import Image


class GermanyDataset(Dataset):
    def __init__(self, folder_path, transforms=None, size=[640, 640], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.folder_path = folder_path
        self.transforms = transforms
        self.size = size
        self.mean = mean
        self.std = std

        google_dir = os.path.join(folder_path, "google")
        ign_dir = os.path.join(folder_path, "ign")

        self.google_images = os.listdir(os.path.join(google_dir, "img"))
        self.ign_images = os.listdir(os.path.join(ign_dir, "img"))

        self.dataset = self.google_images + self.ign_images

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        suffix = "google" if self.dataset[idx] in self.google_images else "ign"

        # Get the image from the image folder
        image_path = os.path.join(self.folder_path, suffix, "img", self.dataset[idx])
        image = Image.open(image_path).convert("RGB")

        # Check if there is a mask in the mask folder
        try:
            mask_path = os.path.join(
                self.folder_path, suffix, "mask", self.dataset[idx]
            )
            mask = Image.open(mask_path).convert("L")
            mask = mask.point(lambda p: p > 0 and 1)
        except FileNotFoundError:
            # If no mask is found generate an empty mask
            mask = Image.new("L", image.size)

        image = F.to_image(image)
        image = F.to_dtype(image, dtype=torch.float32, scale=True)
        image = F.resize(image, size=self.size)
        image = F.normalize(image, mean=self.mean, std=self.std)

        mask = F.to_image(mask)
        mask = F.to_dtype(mask, dtype=torch.int, scale=False)
        mask = F.resize(mask, size=self.size, interpolation=F.InterpolationMode.NEAREST)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
