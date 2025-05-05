from django.shortcuts import render
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from .models import WasteClassifier, WastePrediction
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Dynamically resolve the path to the model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'waste_model.pth')

# Load trained model once
model = WasteClassifier()
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except FileNotFoundError:
    logger.error(f"Model file not found at {model_path}. Please train the model first.")
    model = None
except EOFError:
    logger.error(f"Model file at {model_path} is corrupted. Please retrain the model.")
    model = None

# Waste categories (match your training folders)
class_names = ['Glass', 'Metal', 'Organic', 'Paper', 'Plastic', 'Fabric', 'Unknown']

# Material and recyclability map
material_map = {
    'Plastic': ('Polyethylene (PE)', True),
    'Paper': ('Cellulose', True),
    'Metal': ('Aluminum or Steel', True),
    'Organic': ('Biodegradable (e.g., food waste)', True),
    'Glass': ('Silica (SiO₂)', True),
    'Fabric': ('Fabric (Cotton, Polyester)', False)
}

def home(request):
    """Render the home page."""
    return render(request, 'home.html')

def classify(request):
    if request.method == 'POST':
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return render(request, 'home.html', {'error': 'No image uploaded.'})
            
            # Validate image type
            try:
                image = Image.open(image_file).convert('RGB')
            except:
                return render(request, 'home.html', {'error': 'Uploaded file is not an image. Please upload a valid image.'})
            
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            input_tensor = transform(image).unsqueeze(0)

            # Predict using model
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, pred = torch.max(probabilities, 1)
                label = class_names[pred.item()]
                confidence_score = float(confidence.item()) * 100

            # Material and recyclable
            material_type, recyclable = material_map.get(label, ("Unknown Material", False))

            # Save prediction
            WastePrediction.objects.create(
                image=image_file,
                label=label,
                confidence=confidence_score,
                material_type=material_type,
                recyclable=recyclable
            )

            prediction = {
                'label': label,
                'confidence': round(confidence_score, 2),
                'recyclable': recyclable,
                'material_type': material_type
            }

            return render(request, 'home.html', {'prediction': prediction})

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return render(request, 'home.html', {'error': str(e)})

    return render(request, 'home.html')
