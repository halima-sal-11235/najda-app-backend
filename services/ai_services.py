# services/ai_service.py
from PIL import Image
from .yolo_service import count_objects
from .clip_service import accident_score

def analyze_accident_image(image: Image.Image) -> dict:
    """
    Runs YOLOv8 object detection and CLIP classification on a PIL image instance.
    Returns structured results for database storage and API response.
    """
    # 1. Run YOLOv8 detection
    vehicles, people = count_objects(image, conf=0.4)

    # 2. Run CLIP zero-shot classification
    score = accident_score(image)

    return {
        "vehicles_count": vehicles,
        "people_count": people,
        "accident_score": score,
        "is_confirmed": score >= 0.70
    }