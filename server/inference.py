from torch import float32
import torchvision.transforms.v2 as transforms

import numpy as np
import cv2
from PIL import Image

import torch
from losses import LossJaccard

from models.architectures import DeepLabModel
from models.base import BaseModel

segmentation_model = None


def load_model():
    global segmentation_model

    model = DeepLabModel(2, backbone="resnet152")
    model.load_state_dict(torch.load("server/segmentation_model.pth"))
    segmentation_model = BaseModel(model, LossJaccard(), None)
    segmentation_model.eval()


def clean_up_model():
    global segmentation_model

    segmentation_model = None


def masks_to_polygons(mask):
    mask = mask.cpu().numpy().astype("uint8")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(approx)

    return polygons


def find_polygon_centers(polygons):
    centers = []
    for polygon in polygons:
        moments = cv2.moments(polygon)
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        centers.append(center)

    return centers


def find_polygon_boundaries(polygons):
    """Given the polygons list find the lower left and upper right corners.

    Args:
        polygons (_type_): _description_
    """

    extreme_points_per_polygon = []

    # Iterate through each polygon
    for polygon in polygons:
        # Calculate the top left and bottom right points for the current polygon
        top_left = np.min(polygon, axis=0)[0]
        bottom_right = np.max(polygon, axis=0)[0]

        # Append the extreme points to the list
        extreme_points_per_polygon.append((top_left, bottom_right))

    return extreme_points_per_polygon


def plot_mask(mask, polygons, centers):
    from matplotlib import pyplot as plt

    plt.imshow(mask[0].cpu().numpy(), cmap="gray")

    # show the polygons
    for polygon in polygons:
        plt.plot(polygon[:, 0, 0], polygon[:, 0, 1], "r-")

    # show the centers
    for center in centers:
        plt.plot(center[0], center[1], "ro")

    plt.show()


def segmentation_inference(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(float32, scale=True),
            transforms.Resize(
                (640, 640), interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = transform(image).unsqueeze(0)

    mask = segmentation_model(image).argmax(1)

    polygons = masks_to_polygons(mask.squeeze(0))
    centers = find_polygon_centers(polygons)
    boundaries = find_polygon_boundaries(polygons)

    return polygons, centers, boundaries
