from django.urls import path
from . import views
from rest_framework_simplejwt.views import TokenRefreshView
from django.contrib import admin
from django.urls import path
from . import views
from .views import ResourceView, ResourceByIdView
from .views import RegisterView, LoginView, TranscriptAnalysisView, QueryView, ChatSessionView, WaitListView

urlpatterns = [
    path('register/', views.RegisterView.as_view(), name='register'),
    path('login/', views.LoginView.as_view(), name='login'),
    path('analyze-transcript/', views.TranscriptAnalysisView.as_view(), name='analyze-transcript'),
    path('query/', views.QueryView.as_view(), name='query'),
    path('chat-sessions/', ChatSessionView.as_view(), name='chat_sessions'),  # New endpoint
    path('wait-list/', WaitListView.as_view(), name='wait_list'),  # New endpoint
    path('analyze-document/', views.DocumentAnalysisView.as_view(), name='analyze-document'),
    path('resources/', ResourceView.as_view(), name='resources'),
    path('resources/by-resource-id/<str:resource_id>/', ResourceByIdView.as_view(), name='resource-by-id'),
    path('resources/<str:resource_id>/toc/', views.TableOfContentsView.as_view(), name='document-toc'),
]