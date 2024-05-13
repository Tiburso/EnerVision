import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2.functional as F

import os


class SolarDKDataset(Dataset):
    """
    A dataset class for loading images and masks of the Solar DK dataset
    
    This class loads images and masks of the Solar DK dataset to make them suitable for image segmentation.
    The images and mask are by default loaded as 
    """
    def __init__(
        self, image_dir, image_transform=None, mask_transform=None, total_samples: int = None):
        # Get all files in the image directory either in the positive or negative folders
        self.positive_files = os.listdir(os.path.join(image_dir, "positive"))
        self.negative_files = os.listdir(os.path.join(image_dir, "negative"))

        if total_samples is not None:
            self.negative_files = self.negative_files[
                : total_samples - len(self.positive_files)
            ]

        # Concat both lists
        self.images = self.positive_files + self.negative_files

        # Set the image directory, image transformations, and mask transformations
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

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

        image = F.to_image(image)
        image = F.to_dtype(image, dtype=torch.float32, scale=True)

        if self.image_transform:
            image = self.image_transform(image)
        
        mask = F.to_image(mask)
        mask = F.to_dtype(mask, dtype=torch.float32, scale=False)

        if self.mask_transform:
            mask = self.mask_transform(mask)
    
        return image, mask
