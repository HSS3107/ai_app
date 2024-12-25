from django.contrib import admin
from django.urls import path, include
from app.views import api_root

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', api_root, name='home'),  # Root URL shows API welcome
    path('api/', include('app.urls')),  # Include all app URLs under /api/
]