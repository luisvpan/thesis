import torch
from ultralytics.models import YOLO

print(torch.cuda.is_available())

# torch.cuda.set_device(0)

# model = YOLO("yolo11n.pt")

# results = model.train(data="./datasets/mnist_detection/data.yaml", epochs=20, imgsz=1080, batch=16, device='gpu')
