import torch
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from PIL import Image

from models.base import BaseModel

from dataloaders.solar_dk_dataset import SolarDKDataset
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from torchmetrics.functional import (
    jaccard_index,
    accuracy,
    precision,
    recall,
    f1_score,
    dice,
)

test_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"

model = BaseModel.load_from_checkpoint(
    "lightning_logs/version_206347/checkpoints/epoch=21-step=7370.ckpt"
)

test_dataset = SolarDKDataset(test_folder)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
trainer = Trainer()

trainer.test(model, test_loader)

task = "multiclass"
num_classes = 2

# Estimate the model on the test set
model.eval()
with torch.no_grad():
    image, label = next(iter(test_loader))

    idx = 3
    output = model(image)[idx]
    label = label[idx]

    # sigmoid_predictions = torch.sigmoid(output)
    # output_predictions = (torch.sigmoid(output) > 0.1).float().squeeze()
    output_predictions = output.argmax(dim=0).float().squeeze()
    label = label.argmax(dim=0).float().squeeze()

    print(f"Dice Score: {dice(output_predictions, label.int())}")
    print(
        f"Jaccard Index: {jaccard_index(output_predictions, label, task=task, num_classes=num_classes)}"
    )
    print(
        f"Accuracy: {accuracy(output_predictions, label, task=task, num_classes=num_classes)}"
    )
    print(
        f"Precision: {precision(output_predictions, label, task=task, num_classes=  num_classes)}"
    )
    print(
        f"Recall: {recall(output_predictions, label, task=task, num_classes=num_classes)}"
    )
    print(
        f"F1 Score: {f1_score(output_predictions, label, task=task, num_classes=num_classes)}"
    )

    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap="gray")
    plt.title("Label")

    plt.subplot(1, 3, 3)
    r = Image.fromarray(output_predictions.byte().cpu().numpy())
    plt.imshow(r, cmap="gray")
    plt.title("Output")

    plt.show()
