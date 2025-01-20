from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.yaml")

# Train the model on the COCO8 example dataset for 3 epochs
results = model.train(data="config.yaml", epochs=1)