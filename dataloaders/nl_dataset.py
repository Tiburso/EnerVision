from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import torchvision.transforms.v2.functional as F
import os

from roboflow import Roboflow
from dotenv import load_dotenv


class CocoSegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None, normalize=False, download=False):
        if download:
            self.download()

        self.image_dir = image_dir

        annotation_file = os.path.join(image_dir, "_annotations.coco.json")
        self.coco = COCO(annotation_file)

        self.image_ids = self.coco.getImgIds()
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        mask = self.create_mask(image_id, image_info)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

            if self.normalize:
                image = F.normalize(
                    image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )

        return image, mask

    def create_mask(self, image_id, image_info):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = Image.new("L", (image_info["width"], image_info["height"]))
        for ann in anns:
            if "segmentation" in ann:
                ImageDraw.Draw(mask).polygon(ann["segmentation"][0], outline=1, fill=1)
        return mask

    def download(self):
        load_dotenv()
        DATASET_KEY = os.getenv("API_KEY")

        if DATASET_KEY is None:
            raise ValueError("API_KEY not found in environment variables")

        if not os.path.exists(r"data/NL-Solar-Panel-Seg-1"):
            rf = Roboflow(api_key=DATASET_KEY)
            project = rf.workspace("electasolar").project("nl-solar-panel-seg")
            version = project.version(1)
            version.download("coco-segmentation")
