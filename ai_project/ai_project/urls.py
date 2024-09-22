from django.contrib import admin
from django.urls import path, include
from app import views

urlpatterns = [
    path('', views.home, name='home'),  # Root URL
    path('admin/', admin.site.urls),
    path('api/', include('app.urls')),  # Assuming your app is named 'app'
    path('', include('app.urls')),  # This will include the app URLs at the root level as well
    path('admin/', admin.site.urls),
    path('api/', include('app.urls')),  # This will include the app-level URLs
    path('login/', views.login_view, name='login'),
    path('login-page/', views.login_view, name='login-page'),
    path('register/', views.register_view, name='register'),
    path('register-page/', views.register_view, name='register-page'),
    # path('transcript-query/', views.transcript_query_view, name='transcript-query'),
]
