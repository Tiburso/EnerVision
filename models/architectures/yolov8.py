from ultralytics import YOLO

class Yolov8(BaseModel):
    def __init__(self, num_classes):
        super().__init__()

        self.model = YOLO('yolov8n-seg.pt')

    def forward(self, x):
        return self.model(x)