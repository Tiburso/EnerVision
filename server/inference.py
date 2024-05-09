from torch import float32
from torchvision.ops import masks_to_boxes
import torchvision.transforms.v2 as transforms

import cv2
from PIL import Image

from models.base import BaseModel

segmentation_model = BaseModel.load_from_checkpoint("server/segmentation_model.ckpt")
segmentation_model.eval()


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


def plot_mask(mask, bbox, polygons, centers):
    from matplotlib import pyplot as plt

    plt.imshow(mask[0].cpu().numpy(), cmap="gray")
    # show the bounding boxes
    for box in bbox:
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color="red")

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

    bbox = masks_to_boxes(mask)
    polygons = masks_to_polygons(mask.squeeze(0))
    centers = find_polygon_centers(polygons)

    print(polygons)
    plot_mask(mask, bbox, polygons, centers)

    return mask
