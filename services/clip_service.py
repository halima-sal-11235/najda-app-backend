import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

# The model loads ONCE when the server starts, not on every request
_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

ACCIDENT_PROMPTS = [
    "a photo of a car accident on the road",
    "a photo of a car crash with damaged vehicles",
    "a photo of an overturned car on the road",
]
NORMAL_PROMPTS = [
    "a photo of a normal road with traffic",
    "a photo of an empty road",
    "a photo of cars driving normally",
]
LABELS = ACCIDENT_PROMPTS + NORMAL_PROMPTS


@torch.no_grad()
def accident_score(image: Image.Image) -> float:
    """Returns a number between 0 and 1: how close the image is to 'accident'."""
    inputs = _processor(text=LABELS, images=image,
                        return_tensors="pt", padding=True).to(device)
    probs = _model(**inputs).logits_per_image.softmax(dim=1)[0]
    return float(probs[: len(ACCIDENT_PROMPTS)].sum())