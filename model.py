import torch.nn as nn
from torchvision import models

def get_model(pretrained=True, num_classes=2, dropout=0.3):
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(512, num_classes)
    )
    return model
