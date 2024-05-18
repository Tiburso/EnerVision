import torch
from models.base import BaseModel
import torchvision.transforms.v2 as transforms
from dataloaders.solar_dk_dataset import SolarDKDataset
from dataloaders.nl_dataset import CocoSegmentationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from losses import CombinedLoss
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss

from PIL import Image

from torchmetrics.functional import (
    jaccard_index,
    accuracy,
    precision,
    recall,
    f1_score,
    dice,
)

from sklearn.metrics import jaccard_score, precision_score

import logging

logging.basicConfig(level=logging.DEBUG)


class LossJaccard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = JaccardLoss(mode="multiclass")

    def forward(self, y_hat, y):
        y = y.argmax(dim=1)
        return self.loss(y_hat, y)


inv_transform = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Resize(
            (640, 640), interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"
test_dataset = SolarDKDataset(test_folder, total_samples=500)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# test_folder = "data/NL-Solar-Panel-Seg-1/test"
# test_dataset = CocoSegmentationDataset(test_folder)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

image = Image.open("data/germany_dataset/google/img/ZZZMR9A1DPOFRX.png").convert("RGB")
image = transform(image).unsqueeze(0)

# model = BaseModel.load_from_checkpoint(
#     "lightning_logs/version_251115/checkpoints/last.ckpt"
# )


def main():
    checkpoint = torch.load(
        "lightning_logs/version_251115/checkpoints/last.ckpt",
        map_location=torch.device("cpu"),
    )
    model = BaseModel.load_from_checkpoint(checkpoint)

    model.eval()
    with torch.no_grad():
        iterator = iter(test_loader)
        image, label = next(iterator)
        output = model(image)

        image = inv_transform(image)
        # label = model.model.target(label)
        # # loss = loss_fn(output, label)

        # label = label.argmax(dim=1)

        # # print(f"Loss: {loss.item()}")
        # print(f"Dice Score: {dice(output, label.int())}")
        # print(f"Jaccard Index: {jaccard_index(output, label, task=task, num_classes=num_classes, average='macro')}")
        # print(f"Precision: {precision(output, label, task=task, num_classes=  num_classes, average='macro')}")
        # print(f"Accuracy: {accuracy(output, label, task=task, num_classes=num_classes)}")
        # print(f"Recall: {recall(output, label, task=task, num_classes=num_classes)}")
        # print(f"F1 Score: {f1_score(output, label, task=task, num_classes=num_classes)}")

        fig, ax = plt.subplots(1, 3, figsize=(20, 10))

        ax[0].imshow(image[0].permute(1, 2, 0))
        ax[0].set_title("Image")

        ax[1].imshow(label.squeeze(), cmap="gray")
        ax[1].set_title("Label")

        ax[2].imshow(output.argmax(dim=1).squeeze(), cmap="gray")
        ax[2].set_title("Prediction")

    fig.savefig("output.png")


if __name__ == "__main__":
    main()
