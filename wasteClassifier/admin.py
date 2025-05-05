# filepath: wasteClassifier/admin.py
from django.contrib import admin
from .models import WastePrediction

@admin.register(WastePrediction)
class WastePredictionAdmin(admin.ModelAdmin):
    list_display = ('label', 'confidence', 'recyclable', 'created_at')

# Register models.
