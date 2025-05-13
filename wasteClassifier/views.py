from django.shortcuts import render
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from .models import WasteClassifier, WastePrediction
import os
import logging
from django.db.models import Count
from datetime import timedelta
from django.utils import timezone
from django.http import JsonResponse

# Set up logging
logger = logging.getLogger(__name__)

# Dynamically resolve the path to the model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'waste_model.pth')

# Load trained model once
model = WasteClassifier()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Waste categories 
class_names = ['Fabric', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']

# Material and recyclability map
material_map = {
    
    'Fabric': ('Fabric (Cotton , Polyester)', False),
    'Glass': ('Silica (SiO₂)', True),  
    'Metal': ('Aluminum or Steel', True),
    'Organic': ('Biodegradable', True),
    'Paper': ('Cellulose', True),
    'Plastic': ('Polyethylene (PE)', True),
}

# Waste category indices
category_indices = {'Fabric': 0, 'Glass': 1, 'Metal': 2, 'Organic': 3, 'Paper': 4, 'Plastic': 5}

def home(request):
    # All-time counts (existing)
    category_counts = (
        WastePrediction.objects.values('label')
        .annotate(count=Count('label'))
        .order_by('-count')
    )
    fixed_labels = ['Fabric', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']
    category_dict = {item['label']: item['count'] for item in category_counts}
    counts = [category_dict.get(label, 0) for label in fixed_labels]

    # Weekly trend counts (last 7 days)
    last_week = timezone.now() - timedelta(days=7)
    trend_data = (
        WastePrediction.objects.filter(created_at__gte=last_week)
        .values('label')
        .annotate(count=Count('label'))
    )
    trend_dict = {item['label']: item['count'] for item in trend_data}
    weekly_counts = [trend_dict.get(label, 0) for label in fixed_labels]
    weekly_labels = fixed_labels  # Always keep order

    context = {
        'labels': fixed_labels,
        'counts': counts,
        'weekly_labels': weekly_labels,
        'weekly_counts': weekly_counts,
    }
    return render(request, 'home.html', context)

def classify(request):
    fixed_labels = ['Fabric', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']

    if request.method == 'POST':
        try:
            image_file = request.FILES.get('waste_image')
            if not image_file:
                # Also pass trends data on error
                category_counts = (
                    WastePrediction.objects.values('label')
                    .annotate(count=Count('label'))
                    .order_by('-count')
                )
                category_dict = {item['label']: item['count'] for item in category_counts}
                counts = [category_dict.get(label, 0) for label in fixed_labels]

                last_week = timezone.now() - timedelta(days=7)
                trend_data = (
                    WastePrediction.objects.filter(created_at__gte=last_week)
                    .values('label')
                    .annotate(count=Count('label'))
                )
                trend_dict = {item['label']: item['count'] for item in trend_data}
                weekly_counts = [trend_dict.get(label, 0) for label in fixed_labels]

                return render(request, 'home.html', {
                    'error': 'No image uploaded.',
                    'labels': fixed_labels,
                    'counts': counts,
                    'weekly_counts': weekly_counts,
                })
            
            # Validate image type
            try:
                image = Image.open(image_file).convert('RGB')
            except:
                category_counts = (
                    WastePrediction.objects.values('label')
                    .annotate(count=Count('label'))
                    .order_by('-count')
                )
                category_dict = {item['label']: item['count'] for item in category_counts}
                counts = [category_dict.get(label, 0) for label in fixed_labels]

                last_week = timezone.now() - timedelta(days=7)
                trend_data = (
                    WastePrediction.objects.filter(created_at__gte=last_week)
                    .values('label')
                    .annotate(count=Count('label'))
                )
                trend_dict = {item['label']: item['count'] for item in trend_data}
                weekly_counts = [trend_dict.get(label, 0) for label in fixed_labels]

                return render(request, 'home.html', {
                    'error': 'Uploaded file is not an image. Please upload a valid image.',
                    'labels': fixed_labels,
                    'counts': counts,
                    'weekly_counts': weekly_counts,
                })
            
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
            }

            # All-time counts
            category_counts = (
                WastePrediction.objects.values('label')
                .annotate(count=Count('label'))
                .order_by('-count')
            )
            category_dict = {item['label']: item['count'] for item in category_counts}
            counts = [category_dict.get(label, 0) for label in fixed_labels]

            # Weekly trends (last 7 days)
            last_week = timezone.now() - timedelta(days=7)
            trend_data = (
                WastePrediction.objects.filter(created_at__gte=last_week)
                .values('label')
                .annotate(count=Count('label'))
            )
            trend_dict = {item['label']: item['count'] for item in trend_data}
            weekly_counts = [trend_dict.get(label, 0) for label in fixed_labels]

            context = {
                'prediction': prediction,  # if available
                'labels': fixed_labels,
                'counts': counts,
                'weekly_labels': fixed_labels,
                'weekly_counts': weekly_counts,
            }
            return render(request, 'home.html', context)
        except Exception as e:
            # Also pass trends data on error
            category_counts = (
                WastePrediction.objects.values('label')
                .annotate(count=Count('label'))
                .order_by('-count')
            )
            category_dict = {item['label']: item['count'] for item in category_counts}
            counts = [category_dict.get(label, 0) for label in fixed_labels]

            last_week = timezone.now() - timedelta(days=7)
            trend_data = (
                WastePrediction.objects.filter(created_at__gte=last_week)
                .values('label')
                .annotate(count=Count('label'))
            )
            trend_dict = {item['label']: item['count'] for item in trend_data}
            weekly_counts = [trend_dict.get(label, 0) for label in fixed_labels]

            logger.error(f"Error during prediction: {e}")
            return render(request, 'home.html', {
                'error': str(e),
                'labels': fixed_labels,
                'counts': counts,
                'weekly_counts': weekly_counts,
            })

    # For GET requests, also pass trends data
    category_counts = (
        WastePrediction.objects.values('label')
        .annotate(count=Count('label'))
        .order_by('-count')
    )
    category_dict = {item['label']: item['count'] for item in category_counts}
    counts = [category_dict.get(label, 0) for label in fixed_labels]

    last_week = timezone.now() - timedelta(days=7)
    trend_data = (
        WastePrediction.objects.filter(created_at__gte=last_week)
        .values('label')
        .annotate(count=Count('label'))
    )
    trend_dict = {item['label']: item['count'] for item in trend_data}
    weekly_counts = [trend_dict.get(label, 0) for label in fixed_labels]

    return render(request, 'home.html', {
        'labels': fixed_labels,
        'counts': counts,
        'weekly_counts': weekly_counts,
    })

def weekly_trends(request):
    # Get data from the last 7 days
    one_week_ago = timezone.now() - timedelta(days=7)

    # Count each label's frequency
    weekly_data = WastePrediction.objects.filter(created_at__gte=one_week_ago) \
        .values('label') \
        .annotate(count=Count('label'))

    # Prepare labels and counts for Chart.js
    weekly_labels = [entry['label'] for entry in weekly_data]
    weekly_counts = [entry['count'] for entry in weekly_data]

    return render(request, 'home.html', {
        'weekly_labels': weekly_labels,
        'weekly_counts': weekly_counts
    })

def trends_api(request):
    fixed_labels = ['Fabric', 'Glass', 'Metal', 'Organic', 'Paper', 'Plastic']
    category_counts = (
        WastePrediction.objects.values('label')
        .annotate(count=Count('label'))
    )
    category_dict = {item['label']: item['count'] for item in category_counts}
    counts = [category_dict.get(label, 0) for label in fixed_labels]
    return JsonResponse({'labels': fixed_labels, 'counts': counts})