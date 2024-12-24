from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.decorators import api_view
from django.contrib.auth import authenticate
from . import podcast_utils
import logging
import traceback
from django.conf import settings
import re
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from .models import ChatSession  # Import the ChatSession model
from .serializers import UserSerializer, ChatSessionSerializer, WaitListSerializer  # Ensure you have a serializer for ChatSession
from django.contrib.auth.models import User
# ... (other imports)


logger = logging.getLogger(__name__)

@api_view(['GET'])
def api_root(request):
    return Response({"message": "Welcome to YT Transcript Analyzer API"})

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

class TranscriptAnalysisView(APIView):
    def post(self, request):
        video_url = request.data.get('video_url')
        if not video_url:
            return Response({'error': 'Video URL is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/|.+\?v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
            match = re.search(pattern, video_url)

            if match:
                video_id = match.group(1)
            else:
                return Response({'error': 'Invalid YouTube URL'}, status=status.HTTP_400_BAD_REQUEST)

            vector_store, retrieval_chain, video_info, content_map, summary_and_tags, chat_history = podcast_utils.process_video(video_id)
            
            # Retrieve the full transcript
            transcript = podcast_utils.get_transcript(video_id)
            formatted_transcript = podcast_utils.format_original_transcript(transcript, video_info, video_id) if transcript and video_info else "Transcript not available."
            
            response_data = {
                'message': 'Video processed',
                'video_id': video_id,
                'content_map': content_map or "Content map not available",
                'video_info': video_info or {'title': 'Unknown', 'channel': 'Unknown', 'published_at': 'Unknown', 'duration': 'Unknown', 'captions_type': 'Unknown'},
                'summary_and_tags': summary_and_tags or "Summary and tags not available",
                'transcript': formatted_transcript
            }
            
            if not vector_store or not retrieval_chain:
                response_data['warning'] = 'Vector store or retrieval chain creation failed. Some features may be unavailable.'
            
            # Store the chat_history in the session
            request.session[f'chat_history_{video_id}'] = chat_history
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error processing video ID {video_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({'error': 'Failed to process video', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from .podcast_utils import load_vector_store, setup_llm_chain, process_query, get_video_info

from .podcast_utils import load_vector_store, get_video_info, process_query, setup_llm_chain
from .podcast_utils import load_vector_store, get_video_info, process_query, setup_llm_chain, get_summary


class QueryView(APIView):
    def post(self, request):
        video_id = request.data.get('video_id')
        query = request.data.get('query')
        session_identifier = request.data.get('session_identifier')
        
        if not all([video_id, query, session_identifier]):
            return Response({
                'error': 'Video ID, query, and session identifier are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            vector_store = load_vector_store(video_id)
            if vector_store is None:
                return Response({'error': 'Failed to load vector store for this video'}, 
                              status=status.HTTP_404_NOT_FOUND)
            
            video_info = get_video_info(video_id)
            if video_info is None:
                return Response({'error': 'Failed to get video info'}, 
                              status=status.HTTP_404_NOT_FOUND)

            retriever = vector_store.as_retriever()
            retrieval_chain = setup_llm_chain(retriever)
            content_map = get_summary(video_id)

            # Process query with session-specific chat history
            result, _ = process_query(
                retrieval_chain, 
                query, 
                video_id, 
                video_info['title'], 
                content_map,
                session_identifier
            )
            
            return Response({
                'result': result
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response({'error': 'Failed to process query', 'details': str(e)}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatSessionView(APIView):
    def post(self, request):
        logger.info(f"Received chat session request data: {request.data}")
        
        serializer = ChatSessionSerializer(data=request.data)
        if serializer.is_valid():
            try:
                chat_session = serializer.save()
                return Response({
                    'message': 'Chat session created successfully!',
                    'id': chat_session.id
                }, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.error(f"Error creating chat session: {str(e)}")
                return Response({
                    'error': 'Failed to create chat session',
                    'detail': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        logger.error(f"Serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class WaitListView(APIView):
    def post(self, request):
        logger.info(f"Received wait list request data: {request.data}")
        
        serializer = WaitListSerializer(data=request.data)
        
        if serializer.is_valid():
            try:
                wait_lister = serializer.save()
                return Response({
                    'message': 'Wait lister added successfully!',
                    'id': wait_lister.id
                }, status=status.HTTP_201_CREATED)
            except Exception as e:
                logger.error(f"Error creating waitlister: {str(e)}")
                return Response({
                    'error': 'Failed to create waitlister',
                    'detail': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  # Added this return statement