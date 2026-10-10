import torch
import torch.nn as nn
from torchvision import models


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Create ResNet18 model
model = models.resnet18(weights=None)

# Change final layer for 3 classes
model.fc = nn.Linear(model.fc.in_features,3)


# Load trained weights
model.load_state_dict(torch.load("resnet18_cat_dog_human.pth",map_location=device))

# Move model to device
model = model.to(device)


# Evaluation mode
model.eval()