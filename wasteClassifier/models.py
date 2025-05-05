# Django model 
from django.db import models
import torch.nn as nn

class WastePrediction(models.Model):
    image = models.ImageField(upload_to='waste_images/')
    label = models.CharField(max_length=50)
    confidence = models.FloatField()
    material_type = models.CharField(max_length=100, blank=True)
    recyclable = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence}%)"

# PyTorch model 


class WasteClassifier(nn.Module):
    def __init__(self):
        super(WasteClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128), nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)
