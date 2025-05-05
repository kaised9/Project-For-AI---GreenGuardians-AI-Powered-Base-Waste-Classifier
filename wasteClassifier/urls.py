from django.contrib import admin
from django.urls import path
from . import views  # Import your views module
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('classify/', views.classify, name='classify'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
media_path = os.path.join(BASE_DIR, 'media')
print(media_path) 