# 🌱 GreenGuardian: AI-Powered Waste Classifier

GreenGuardian is a Django-based web application that uses artificial intelligence to classify waste images into different categories like **plastic, paper, organic, or e-waste**. It helps users identify types of waste easily to promote better waste management and recycling.


## 💡 Features

- 🖼 Upload an image of waste
  
- 🤖 AI model classifies it (e.g., plastic, paper, organic, etc.)
  
- 📊 Displays the prediction result
  
- 🌐 Simple and clean user interface


## 🛠 Tech Stack

- **Backend**: Python, Django
- **Frontend**: HTML, CSS
- **AI Model**: Trained machine learning model for waste classification
- **Database**: SQLite (default for Django)

---

## 📁 Project Structure

GreenGuardian/
├── GreenGuardian/ # Main Django settings and URLs
├── wasteClassifier/ # Main app
│ ├── templates/ # HTML files
│ ├── static/ # CSS, JS, image files
│ ├── views.py # App logic
│ ├── urls.py # App URL routes
│ └── models.py # Model (if any)
├── media/ # Uploaded images
├── db.sqlite3 # Database
└── manage.py # Django management script



#  Run the Server

python manage.py runserver

@ Visit: http://127.0.0.1:8000/ in  browser.

# 🖼 Uploading Waste Image
Go to homepage

Choose a waste image to upload

Click "Predict"

View classification result below the image

# 🧠 AI Model
This project uses a simple image classification model trained to identify waste categories. The model is loaded in Django and makes a prediction based on the uploaded image.

# 📸 Media Files
Uploaded images are saved in the /media/ folder. During development, Django serves these files when DEBUG=True.

Ensure you have this in urls.py:

from django.conf import settings
from django.conf.urls.static import static

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# 📃 License

This project is for educational purposes only.

# Contact 

Name  : Md. Kaised Mollick 
Gmail : a.r.kaised.7698@gmail.com
Green University Of Bangladesh
