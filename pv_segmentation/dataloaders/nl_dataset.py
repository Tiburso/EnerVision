from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import torchvision.transforms.v2.functional as F
import os
import torch


class NLSegmentationDataset(Dataset):
    """
    Dataset class for loading and processing images and their segmentation masks from the
    COCO format annotations in the NL Solar Panel segmentation dataset.
    """

    def __init__(
        self,
        image_dir,
        transforms=None,
        size=[640, 640],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ):
        """
        Initializes the NL Solar Panel Segmentation dataset.

        Args:
            image_dir (str): The directory to the images and annotation file.
            transforms (str, optional): Transformations applied to the images and masks. Default is None.
            size (list): Size for images and masks as [width, height]. Default is [640, 640].
            mean (list): Mean values for image normalization. Default is [0.485, 0.456, 0.406].
            std (list): Standard deviation values for image normalization. Default is [0.229, 0.224, 0.225].
        """
        self.image_dir = image_dir
        self.transforms = transforms
        self.size = size
        self.mean = mean
        self.std = std

        # Read the annotation file, and get IDs of images
        annotation_file = os.path.join(image_dir, "_annotations.coco.json")
        self.coco = COCO(annotation_file)
        self.image_ids = self.coco.getImgIds()

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Gets the images and the corresponding segmentation mask at specified index.

        Args:
            index (int): The index of the image to g.

        Returns:
            tuple: A tuple containing the image and the corresponding segmentation mask.
        """
        # Create mask from annotation
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        mask = self.create_mask(image_id, image_info)
        mask = mask.point(lambda p: p > 0 and 1)

        # Convert image to tensor
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

    def create_mask(self, image_id, image_info):
        """
        Creates a segmentation mask for image.

        Args:
            image_id (int): The ID of the image to create a mask for.
            image_info (dict): A dictionary containing information about the image.

        Returns:
            Image: A segmentation mask for image
        """
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = Image.new("L", (image_info["width"], image_info["height"]))
        for ann in anns:
            if "segmentation" in ann:
                ImageDraw.Draw(mask).polygon(ann["segmentation"][0], outline=1, fill=1)

        return mask
