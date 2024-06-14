import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2.functional as F
import os


class SolarDKDataset(Dataset):
    """
    Dataset class for loading and processing images and their segmentation masks from the 
    SolarDK UV panel segmentation dataset.

    Args:
        image_dir (str): The directory to the images and annotation file.
        transforms (str, optional): Transformations applied to the images and masks. Default is None.
        size (list): Size for images and masks as [width, height]. Default is [640, 640].
        mean (list): Mean values for image normalization. Default is [0.485, 0.456, 0.406].
        std (list): Standard deviation values for image normalization. Default is [0.229, 0.224, 0.225].
    """
    def __init__(self, image_dir, transforms=None, size=[640, 640], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        # Set the image directory, image transformations, and mask transformations
        self.image_dir = image_dir
        self.transforms = transforms
        self.size = size
        self.mean = mean
        self.std = std

        # Get all files in the image directory either in the positive or negative folders
        self.positive_files = os.listdir(os.path.join(image_dir, "positive"))
        self.negative_files = os.listdir(os.path.join(image_dir, "negative"))

        num_positive = len(self.positive_files)

        self.positive_files = self.positive_files[:num_positive]
        self.negative_files = self.negative_files[:num_positive]

        # Concat both lists
        self.images = self.positive_files + self.negative_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Gets the images and the corresponding segmentation mask at specified index.

        Returns:
            tuple: A tuple containing the image and the corresponding segmentation mask.
        """
        prefix = "positive" if self.images[index] in self.positive_files else "negative"
        image_path = os.path.join(self.image_dir, prefix, self.images[index])
        image = Image.open(image_path).convert("RGB")
        # If the image is in the positive folder, the mask is gonna be in the mask folder
        if prefix == "positive":
            mask_path = os.path.join(self.image_dir, "mask", self.images[index])
            mask = Image.open(mask_path).convert("L")
            mask = mask.point(lambda p: p > 0 and 1)
        # Create an empty mask if the image is in the negative folder
        else:
            mask = Image.new("L", image.size)

        # Preprocess image
        image = F.to_image(image)
        image = F.to_dtype(image, dtype=torch.float32, scale=True)
        image = F.resize(image, size=self.size)
        image = F.normalize(image, mean=self.mean, std=self.std)

        # Preprocess mask
        mask = F.to_image(mask)
        mask = F.to_dtype(mask, dtype=torch.int, scale=False)
        mask = F.resize(mask, size=self.size, interpolation=F.InterpolationMode.NEAREST)

        # Additional transformation
        if self.transforms:
            image, mask = self.transforms(image, mask)
    
        return image, mask
