import cv2
import numpy as np
from shapely.geometry import Polygon
from rasterio.features import geometry_mask, rasterize
from torchvision.ops import masks_to_boxes
from torchvision.tv_tensors import Mask
import torch


def polygons_to_masks(polygons, size=832):
    # Create a mask per polygon
    masks = [rasterize([poly], out_shape=(size, size)) for poly in polygons]
    return [Mask(torch.tensor(mask, dtype=torch.bool)) for mask in masks]


# Convert the mask to a polygon
def mask_to_polygons(mask):
    mask = mask.astype(np.uint8)

    # Reshape into size x size x 1
    mask = mask.reshape(mask.shape[1], mask.shape[1], 1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        polygons.append(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]))
    return polygons


def polygons_to_bounding_boxes(polygons):
    boxes = []
    for polygon in polygons:
        x, y, w, h = cv2.boundingRect(np.array(polygon.exterior.coords, dtype=np.int32))
        boxes.append([x, y, x + w, y + h])
    return boxes
