from typing import Tuple, List

from torch import float32
import torchvision.transforms.v2 as transforms

import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
import pickle

# from losses import LossJaccard

# from models.architectures import DeepLabModel
from models.base import BaseModel

from server.energy_prediction_model import EnergyPredictionModel

segmentation_model = None
energy_prediction_model = None


def load_models():
    """Load the machine learning models - Both these models need to be previously trained 
    and saved in the same directory as this script.
    """
    
    global segmentation_model, energy_prediction_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the segmentation model
    segmentation_model = BaseModel.load_from_checkpoint("segmentation_model.ckpt")
    segmentation_model.eval()
    segmentation_model.to(device)

    # Load the dataset values
    with open("dataset_values.pkl", "rb") as f:
        dataset_values = pickle.load(f)

    # Load the energy prediction model
    # dynamic feature size =5, static feature_size =3. Hidden /fc_size is 8/128
    energy_prediction_model = EnergyPredictionModel(
        dynamic_feature_size=5,
        static_feature_size=3,
        hidden_size=8,
        fc_size=128,
        dataset_values=dataset_values,
    )
    energy_prediction_model.load_state_dict(torch.load("energy_prediction_model.pth"))
    energy_prediction_model.eval()
    energy_prediction_model.to(device)


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

    mask = mask.cpu().numpy().astype(np.uint8)

    # Put the mask in grey scale
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
    mask = mask.cpu().numpy().astype(np.uint8)

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

    probs = torch.sigmoid(segmentation_model(image))
    mask = (probs > 0.5).int()

    mask = mask.squeeze(0).squeeze(0)

    pvtypes = infer_panel_types(image, mask)
    polygons = masks_to_polygons(mask)
    centers = find_polygon_centers(polygons)

    return polygons, centers, pvtypes


def energy_prediction(df: pd.DataFrame) -> List[List[int]]:
    """Predict the energy output for each day based on the given dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the desired features, each row is a day

    Returns:
        List[List[int]]: The predicted output for each hour for two days
    """

    # Column order
    # sample dynamics = Bx24x5 and sample static is Bx3
    dynamic_cols = [
        "temperature_sequence",
        "wind_speed_sequence",
        "dni_sequence",
        "dhi_sequence",
        "global_irradiance_sequence",
    ]
    static_cols = ["tilt", "azimuth", "module_type"]

    module_type_map = {
        "monocrystalline": 0,
        "polycrystalline": 1,
    }

    # Put the dynamic columns in a tensor
    sample_dynamic = torch.tensor(df[dynamic_cols].values, dtype=torch.float32)
    # size 2x24x5
    sample_dynamic = sample_dynamic.view(-1, 24, 5)

    # Put the static columns in a tensor
    df["module_type"] = df["module_type"].map(module_type_map)
    sample_static = torch.tensor(df[static_cols].values, dtype=torch.float32)
    # size 2x3
    sample_static = sample_static.view(-1, 3)[:2]

    predictions: torch.Tensor = energy_prediction_model.predict(
        sample_dynamic, sample_static
    )

    return predictions.tolist()
