from typing import Tuple, List

from torch import float32
import torchvision.transforms.v2 as transforms

import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
from losses import LossJaccard

from models.architectures import DeepLabModel
from models.base import BaseModel

from energy_prediction import EnergyPredictionPL, normalize_features

segmentation_model = None
energy_prediction_model = None


def load_models():
    global segmentation_model, energy_prediction_model

    model = DeepLabModel(2, backbone="resnet152")
    model.load_state_dict(torch.load("segmentation_model.pth"))
    segmentation_model = BaseModel(model, LossJaccard(), None)
    segmentation_model.eval()

    energy_prediction_model = EnergyPredictionPL.load_from_checkpoint(
        "energy_prediction_model.ckpt"
    )
    energy_prediction_model.eval()


def clean_up_models():
    global segmentation_model, energy_prediction_model

    segmentation_model = None
    energy_prediction_model = None


def masks_to_polygons(mask: torch.Tensor) -> list:
    """Convert the segmentation mask to polygons.

    Args:
        mask (torch.Tensor): The segmentation mask.

    Returns:
        list: A list of polygons.
    """

    mask = mask.cpu().numpy().astype("uint8")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the polygon area is less than 100, then it is not a valid polygon
        if cv2.contourArea(approx) < 100:
            continue

        # If the polygon has more than 3 points, then it is a valid polygon
        if approx.shape[0] >= 3:
            polygons.append(approx)

    return polygons


def find_polygon_centers(polygons: list) -> list:
    """Find the center of the polygons.

    Args:
        polygons (list): A list of polygons.

    Returns:
        list: A list of centers.
    """

    centers = []
    for polygon in polygons:
        moments = cv2.moments(polygon)
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        centers.append(center)

    return centers


def infer_panel_types(image: Image.Image, mask: torch.Tensor) -> list:
    """Infer the panel types based on the given image, mask and centers.
    Panel types can either be monocrytalline or polycrystalline.

    Args:
        image (Image): The image to infer the panel types from.
        mask (torch.Tensor): The segmentation mask.

    Returns:
        list: The inferred panel types.
    """
    black_threshold = 30  # Lower value for black
    blue_threshold = 200  # Higher saturation and value for blue

    # Find the contours of the mask
    mask = mask.cpu().numpy().astype("uint8")

    # Convert the image to HSV color space
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB)
    hsv_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2HSV)

    # Find the countours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask with the same size as the image
    panel_types = []
    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI) from the HSV image
        roi = hsv_image[y : y + h, x : x + w]

        # Calculate the average color of the ROI
        avg_color = np.mean(roi, axis=(0, 1))

        # Check if the average color is black
        if avg_color[2] < black_threshold:
            panel_types.append("monocrystalline")
        elif avg_color[0] > blue_threshold and avg_color[1] > blue_threshold:
            panel_types.append("polycrystalline")
        else:
            panel_types.append("monocrystalline")

    return panel_types


def segmentation_inference(image: Image.Image) -> Tuple[list, list, list]:
    """Executes the segmentation model on the given image and returns the polygons, centers
    and boundaries.

    Args:
        image (Image.Image): The image to run the segmentation model on.

    Returns:
        Tuple[list, list, list]: The polygons, centers and the pv types
    """

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

    pvtypes = infer_panel_types(image, mask.squeeze(0))
    polygons = masks_to_polygons(mask.squeeze(0))
    centers = find_polygon_centers(polygons)

    return polygons, centers, pvtypes


def energy_prediction(df: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """Predict the energy output for each day based on the given dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the desired features, each row is a day

    Returns:
        List[Tuple[int, int, int]]: The predicted energy output for each day (the gaussian parameters)
    """

    # df = normalize_features(df)

    # # Convert the dataframe to a tensor
    # x = torch.tensor(df.values).float()

    # # Run the model
    # predictions = energy_prediction_model(x)

    # WIP
    df = pd.read_csv("dataset_to_train_model.csv")

    df = df["gaussian_params"].map(lambda x: ", ".join(x.split()))

    return list(map(lambda x: eval(x), df.iloc[:2].values))
