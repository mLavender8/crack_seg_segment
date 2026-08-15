from ultralytics import YOLO

model = YOLO(r"E:\yolo\1\ultralytics-main\best.pt")
results = model(r"E:\yolo\1\ultralytics-main\predictdata\附件1")
results
