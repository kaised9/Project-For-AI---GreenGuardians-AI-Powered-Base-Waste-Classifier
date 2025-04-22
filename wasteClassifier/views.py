from django.shortcuts import render

# Create your views here.

def home(request):
    # Renders the home.html template
    return render(request, 'home.html')

# views.py
def classify(request):
    if request.method == 'POST':
        image = request.FILES['waste_image']

        # Your model prediction logic here
        prediction_label = "Plastic"
        confidence_score = 87
        recyclable = True
        material_type = "Polyethylene (PE)"
        
        prediction = {
            'label': prediction_label,
            'confidence': confidence_score,
            'recyclable': recyclable,
            'material_type': material_type
        }

        return render(request, 'home.html', {'prediction': prediction})

    return render(request, 'home.html')
