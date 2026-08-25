from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from src.model import CIFAR10ResNet18


CHECKPOINT_PATH = Path("checkpoints/model.pt")

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

app = FastAPI(title="CIFAR-10 ResNet-18 API")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None


def load_model():
    global model

    model = CIFAR10ResNet18()

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ]
)


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "status": "ok",
        "model": "cifar10-resnet18",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    try:
        contents = await image.read()

        pil_image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        tensor = transform(pil_image)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(
                outputs,
                dim=1,
            )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        return {
            "predicted_class": CLASS_NAMES[predicted_index],
            "class_index": predicted_index,
            "probabilities": {
                CLASS_NAMES[i]: round(
                    float(probabilities[i]),
                    6,
                )
                for i in range(len(CLASS_NAMES))
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}",
        )