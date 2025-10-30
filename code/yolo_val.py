from ultralytics.models import YOLO

model = YOLO("./runs/detect/train/weights/best.pt'")

results = model.val()
print(results.confusion_matrix.to_df())
