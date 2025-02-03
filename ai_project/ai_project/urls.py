from django.contrib import admin
from django.urls import path, include
from app.views import api_root  # Import api_root view, no need for home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', api_root, name='home'),  # Route root URL to api_root
    path('api/', api_root, name='api-root'),  # API root
    path('api/', include('app.urls')),  # Include app-specific URLs under /api/
]