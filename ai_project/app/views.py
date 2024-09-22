from rest_framework import status, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.shortcuts import render
from .serializers import UserSerializer
from . import podcast_utils
from django.http import HttpResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from django.views.generic import TemplateView
import logging
import traceback

logger = logging.getLogger(__name__)

class RegisterView(APIView):
    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(email=email, password=password)
        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        return Response({'error': 'Invalid Credentials'}, status=status.HTTP_401_UNAUTHORIZED)

class HomeView(TemplateView):
    template_name = 'app/home.html'

def home(request):
    return render(request, 'app/home.html')

def login_view(request):
    return render(request, 'app/login.html')

def register_view(request):
    return render(request, 'app/register.html')

class TranscriptAnalysisView(APIView):
    def post(self, request):
        video_url = request.data.get('video_url')
        if not video_url:
            return Response({'error': 'Video URL is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            video_id = video_url.split("v=")[1]
        except IndexError:
            return Response({'error': 'Invalid YouTube URL'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            vector_store, chain, video_info, summary, keywords = podcast_utils.process_video(video_id)
            
            response_data = {
                'message': 'Video processed',
                'video_id': video_id,
            }
            
            if summary:
                response_data['summary'] = summary
            if keywords:
                response_data['keywords'] = keywords
            if video_info:
                response_data['video_info'] = {
                    'title': video_info.get('title', 'Unknown'),
                    'channel': video_info.get('channel', 'Unknown'),
                    'published_at': video_info.get('published_at', 'Unknown'),
                    'duration': video_info.get('duration', 'Unknown'),
                    'captions_type': 'Full transcript analysis'
                }
            
            if not vector_store or not chain:
                response_data['warning'] = 'Vector store or chain creation failed. Some features may be unavailable.'
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error processing video ID {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({'error': 'Failed to process video', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class QueryView(APIView):
    def post(self, request):
        video_id = request.data.get('video_id')
        query = request.data.get('query')
        
        if not video_id or not query:
            return Response({'error': 'Video ID and query are required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Fetch video info to get the title
            video_info = podcast_utils.get_video_info(video_id)
            if not video_info:
                return Response({'error': 'Failed to fetch video info'}, status=status.HTTP_400_BAD_REQUEST)
            
            video_title = video_info['title']
            
            # Initialize necessary components
            embeddings = podcast_utils.create_embeddings()
            index = podcast_utils.initialize_pinecone()
            docsearch = podcast_utils.PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
            chain = podcast_utils.setup_llm_chain()
            
            # Process the query
            result = podcast_utils.process_query(docsearch, chain, query, video_id, video_title)
            
            return Response({'result': result}, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error processing query for video_id {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': 'Failed to process query',
                'details': str(e),
                'trace': traceback.format_exc()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)