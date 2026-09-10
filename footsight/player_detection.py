import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

PERSON_LABEL = 1  # COCO category index for "person" in torchvision's label convention (0 = background)


def load_model():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model


def detect_players(
    image_path: str,
    model,
    confidence: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(tensor)[0]

    boxes = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if label.item() == PERSON_LABEL and score.item() >= confidence:
            x1, y1, x2, y2 = box.tolist()
            boxes.append((x1, y1, x2, y2))
    return boxes
