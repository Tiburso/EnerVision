import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as F

import os


class SolarDKDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Get all files in the image directory either in the positive or negative folders
        self.positive_files = os.listdir(os.path.join(image_dir, "positive"))
        self.negative_files = os.listdir(os.path.join(image_dir, "negative"))

        # Concat both lists
        self.images = self.positive_files + self.negative_files

        # Set the image directory
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        prefix = "positive" if self.images[index] in self.positive_files else "negative"
        image_path = os.path.join(self.image_dir, prefix, self.images[index])

        image = Image.open(image_path).convert("RGB")

        # If the image is in the positive folder, the mask is gonna be in the mask folder
        if prefix == "positive":
            mask_path = os.path.join(self.image_dir, "mask", self.images[index])
            mask = Image.open(mask_path).convert("L")
        # Create an empty mask if the image is in the negative folder
        else:
            mask = Image.new("L", image.size)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Convert output into two channels
        mask = torch.cat([1 - mask, mask], dim=0)

        return image, mask
