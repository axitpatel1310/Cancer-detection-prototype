import os
import torch
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Configuration
# -----------------------------

TRAIN_DIR = "dataset/data/train"
TEST_DIR = "dataset/data/test"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_model.pth")

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 5
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# -----------------------------
# Image preprocessing
# -----------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Load dataset
# -----------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

print("\nClasses:")
print(train_dataset.class_to_idx)

print(f"\nTraining images: {len(train_dataset)}")
print(f"Testing images:  {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Load pretrained ResNet18
# -----------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False


# Replace final layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(DEVICE)


# -----------------------------
# Loss + optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training
# -----------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Accuracy: {accuracy:.2f}%"
    )


# -----------------------------
# Evaluation
# -----------------------------

print("\nEvaluating model...\n")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


print("Classification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)


print("Confusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


# -----------------------------
# Save model
# -----------------------------


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes
    },
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")