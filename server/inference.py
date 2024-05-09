from torch import float32
import torchvision.transforms.v2 as transforms
from torchvision.ops import masks_to_boxes

from PIL import Image

from losses import LossJaccard
from models.base import BaseModel

segmentation_model = BaseModel.load_from_checkpoint(
    "server/segmentation_model.ckpt", loss_fn=LossJaccard
)
segmentation_model.eval()


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

    mask = segmentation_model(image).squeeze(0)

    bboxes = masks_to_boxes(mask)

    return mask, bboxes
