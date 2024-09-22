from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenRefreshView

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('analyze-transcript/', views.TranscriptAnalysisView.as_view(), name='analyze-transcript'),
    path('query/', views.QueryView.as_view(), name='query'),
    # path('login-page/', views.login_page, name='login-page'),
    # path('register-page/', views.register_page, name='register-page'),
]
