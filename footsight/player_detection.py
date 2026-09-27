from pathlib import Path

BALL_CLASS = "ball"
GOALKEEPER_ROLE = "goalkeeper"
PLAYER_ROLE = "player"
REFEREE_ROLE = "referee"
PERSON_ROLES = (GOALKEEPER_ROLE, PLAYER_ROLE, REFEREE_ROLE)

DEFAULT_WEIGHTS = Path("weights/roboflow/football-player-detection-v9.pt")

# Real players score 0.86+ on broadcast stills; the losers below this are
# broadcast furniture (a station watermark scored 0.65 on a sample still).
DEFAULT_CONFIDENCE = 0.7
# The stills are ~3000px wide; the model was trained on 1280px imagery and
# loses small, distant players when the frame is squeezed down further.
DEFAULT_IMAGE_SIZE = 1280


def load_model(weights_path: str | Path = DEFAULT_WEIGHTS):
    from ultralytics import YOLO

    return YOLO(str(weights_path))


def detect_players(
    image_path: str,
    model,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Detect the people on the pitch, each tagged with the role the model
    assigned it: "player", "goalkeeper" or "referee".

    The model's own "ball" class is dropped -- footsight locates the ball
    separately, and a ball box projected as a person would put a phantom
    player on the mockup.
    """
    result = model.predict(image_path, conf=confidence, imgsz=image_size, verbose=False)[0]

    detections = []
    for box in result.boxes:
        role = result.names[int(box.cls)]
        if role not in PERSON_ROLES or float(box.conf) < confidence:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(((x1, y1, x2, y2), role))
    return detections
