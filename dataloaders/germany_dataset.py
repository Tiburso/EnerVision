import torch
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from torch.utils.data import Dataset


def load_image_and_labels(file):
    labels = pd.read_csv("germany_dataset/labels/" + file, sep=" ", header=None)
    labels.columns = ["category", "x_center", "y_center", "x_width", "y_width"]

    # Load the image
    image_name = file.replace("txt", "tif")
    image = cv2.imread("germany_dataset/images/" + image_name)

    # image size = 832 x 832
    # The x-center and y-center are in the range [0, 1]

    # Create a list of polygons
    polygons = []
    for i in range(labels.shape[0]):
        x_center = labels.iloc[i, 1]
        y_center = labels.iloc[i, 2]
        x_width = labels.iloc[i, 3]
        y_width = labels.iloc[i, 4]

        x1 = (x_center - x_width / 2) * 832
        x2 = (x_center + x_width / 2) * 832
        y1 = (y_center - y_width / 2) * 832
        y2 = (y_center + y_width / 2) * 832

        polygons.append(Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

    return image, polygons


class GermanyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, polygons = load_image_and_labels(self.dataset[idx])
        mask = polygons_to_mask(polygons)

        return torch.tensor(image, dtype=torch.float32).view(3, 832, 832), torch.tensor(
            mask, dtype=torch.float32
        ).view(1, 832, 832)
