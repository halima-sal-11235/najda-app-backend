from PIL import Image
from ultralytics import YOLO

# yolov8n = the "nano" version: smallest and fastest.
# The weights file downloads automatically on the first run (needs internet).
_model = YOLO("yolov8n.pt")

# Class numbers of the COCO dataset (the dataset YOLO was trained on)
PERSON = 0
VEHICLES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


def count_objects(image: Image.Image, conf: float = 0.4) -> tuple[int, int]:
    """Returns (number of vehicles, number of people).
    conf = confidence threshold: detections below 40% sure are ignored."""
    result = _model.predict(image, conf=conf, verbose=False)[0]
    classes = [int(c) for c in result.boxes.cls.tolist()]
    vehicles = sum(1 for c in classes if c in VEHICLES)
    people = sum(1 for c in classes if c == PERSON)
    return vehicles, people