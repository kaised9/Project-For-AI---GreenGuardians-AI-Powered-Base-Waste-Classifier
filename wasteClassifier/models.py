
# Create Models here.
from django.db import models

class WastePrediction(models.Model):
    image = models.ImageField(upload_to='waste_images/')
    label = models.CharField(max_length=50)
    confidence = models.FloatField()
    material_type = models.CharField(max_length=100, blank=True)
    recyclable = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.label} ({self.confidence}%)"
