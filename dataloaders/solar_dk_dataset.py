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

        # Select only the first 1500 images from the negative folder
        self.negative_files = self.negative_files[:1500]

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

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Convert into torch tensor
        image = F.to_image(image)
        mask = F.to_image(mask)

        # Resize both the image and the mask to 512x512
        image = F.resize(image, (512, 512))
        mask = F.resize(mask, (512, 512), interpolation=TF.InterpolationMode.NEAREST)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        # Apply the normalization
        image = F.to_dtype(image, torch.float32, scale=True)
        image = normalize(image)
        mask = F.to_dtype(mask, torch.float32, scale=True)

        mask = torch.cat([1 - mask, mask], dim=0)

        return image, mask
