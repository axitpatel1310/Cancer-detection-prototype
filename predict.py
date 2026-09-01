import torch

from PIL import Image
from torchvision import transforms, models
from torch import nn


MODEL_PATH = "models/skin_cancer_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load checkpoint
# -----------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]


# -----------------------------
# Recreate model
# -----------------------------

model = models.resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction function
# -----------------------------

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    predicted_class = classes[
        prediction.item()
    ]

    confidence = confidence.item() * 100

    return predicted_class, confidence


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )

    prediction, confidence = predict(
        image_path
    )

    print("\n--------------------------")
    print("AI RESULT")
    print("--------------------------")

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print("--------------------------")