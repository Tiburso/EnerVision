import torch
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from PIL import Image

from models.base import BaseModel

from dataloaders.solar_dk_dataset import SolarDKDataset
from dataloaders.nl_dataset import CocoSegmentationDataset
from dataloaders.germany_dataset import GermanyDataset

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from main import CombinedLoss, LossJaccard

solar_dk_folder = "data/solardk_dataset_neurips_v2/herlev_test/test"
nl_folder = "data/NL-Solar-Panel-Seg-1/test"
germany_folder = "data/germany_dataset"

model = BaseModel.load_from_checkpoint(
    "lightning_logs/version_240425/checkpoints/last.ckpt"
)

# test_dataset = SolarDKDataset(solar_dk_folder) + CocoSegmentationDataset(nl_folder)
test_dataset = GermanyDataset(germany_folder)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
trainer = Trainer()

trainer.test(model, test_loader)
