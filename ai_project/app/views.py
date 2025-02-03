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
from .models import Resource, ResourceType
from .serializers import UserSerializer, ChatSessionSerializer, WaitListSerializer  # Ensure you have a serializer for ChatSession
from django.contrib.auth.models import User
import os
import uuid
from django.conf import settings
from django.core.files.storage import default_storage

from .serializers import (
    UserSerializer, ChatSessionSerializer, WaitListSerializer,
    ResourceSerializer, ResourceDetailSerializer
)
from . import doc_utils

from datetime import datetime

from .processors.smart_document_processor import SmartDocumentProcessor
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

            if not match:
                return Response({'error': 'Invalid YouTube URL'}, status=status.HTTP_400_BAD_REQUEST)
            
            video_id = match.group(1)
            vector_store, retrieval_chain, video_info, content_map, summary_and_tags, chat_history = podcast_utils.process_video(video_id)
            
            # Create or update resource
            resource_data = {
                'resource_id': video_id,
                'resource_type': ResourceType.VIDEO,
                'title': video_info.get('title', f'YouTube Video {video_id}'),
                'content_map': content_map or {},
                'metadata': video_info or {}
            }
            
            resource, created = Resource.objects.update_or_create(
                resource_id=video_id,
                defaults=resource_data
            )
            
            # Retrieve the full transcript
            transcript = podcast_utils.get_transcript(video_id)
            formatted_transcript = podcast_utils.format_original_transcript(transcript, video_info, video_id) if transcript and video_info else "Transcript not available."
            
            response_data = {
                'message': 'Video processed',
                'resource': ResourceSerializer(resource).data,
                'summary_and_tags': summary_and_tags or "Summary and tags not available",
                'transcript': formatted_transcript
            }
            
            if not vector_store or not retrieval_chain:
                response_data['warning'] = 'Vector store or retrieval chain creation failed. Some features may be unavailable.'
            
            request.session[f'chat_history_{video_id}'] = chat_history
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({'error': 'Failed to process video', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from .podcast_utils import load_vector_store, setup_llm_chain, process_query, get_video_info

from .podcast_utils import load_vector_store, get_video_info, process_query, setup_llm_chain
from .podcast_utils import load_vector_store, get_video_info, process_query, setup_llm_chain, get_summary


class QueryView(APIView):
    def post(self, request):
        """Handle POST requests for query processing."""
        # Extract required fields from request
        resource_id = request.data.get('resource_id')
        query = request.data.get('query')
        session_identifier = request.data.get('session_identifier')

        # Validate required fields
        if not all([resource_id, query, session_identifier]):
            return Response({
                'error': 'Resource ID, query, and session identifier are required'
            }, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get resource from database
            resource = Resource.objects.get(resource_id=resource_id)
            
            # Determine which utils to use based on resource type
            utils = podcast_utils if resource.resource_type == ResourceType.VIDEO else doc_utils
            
            # Load vector store and retriever - now handling tuple return value
            vector_store, retriever = utils.load_vector_store(resource_id)
            
            # Check if both vector store and retriever were loaded successfully
            if vector_store is None or retriever is None:
                logger.error(f"Failed to load vector store or retriever for resource ID: {resource_id}")
                return Response(
                    {'error': 'Failed to load vector store or retriever'}, 
                    status=status.HTTP_404_NOT_FOUND
                )

            # Set up the LLM chain using the retriever
            retrieval_chain = utils.setup_llm_chain(retriever)
            
            # Process the query
            result, _ = utils.process_query(
                retrieval_chain,
                query,
                resource_id,
                resource.title,
                resource.content_map,
                session_identifier
            )

            return Response({'result': result}, status=status.HTTP_200_OK)

        except Resource.DoesNotExist:
            logger.error(f"Resource not found: {resource_id}")
            return Response(
                {'error': 'Resource not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return Response({
                'error': 'Failed to process query',
                'details': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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
                    'details': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        logger.error(f"Serializer errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        resource_id = request.query_params.get('resource_id')
        if not resource_id:
            return Response({'error': 'Resource ID is required'}, status=status.HTTP_400_BAD_REQUEST)
            
        try:
            chat_sessions = ChatSession.objects.filter(resource__resource_id=resource_id)
            serializer = ChatSessionSerializer(chat_sessions, many=True)
            return Response(serializer.data)
        except Exception as e:
            logger.error(f"Error retrieving chat sessions: {str(e)}")
            return Response({'error': 'Failed to retrieve chat sessions'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
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
    
    
# class DocumentAnalysisView(APIView):
#     def post(self, request):
#         logger.debug(f"Files in request: {request.FILES}")
#         logger.debug(f"Request data: {request.data}")
        
#         if 'document' not in request.FILES:
#             logger.warning("No document found in request.FILES")
#             return Response({'error': 'Document file is required'}, status=status.HTTP_400_BAD_REQUEST)

#         document = request.FILES['document']
#         try:
#             document_id = str(uuid.uuid4())
#             temp_file_path = handle_uploaded_file(document, document_id)
            
#             try:
#                 vector_store, retrieval_chain, document_info, content_map, summary_and_tags, chat_history = \
#                     doc_utils.process_document(document_id, temp_file_path)
                
#                 # Create or update resource
#                 resource_data = {
#                     'resource_id': document_id,
#                     'resource_type': ResourceType.DOCUMENT,
#                     'title': document_info.get('title', 'Unknown'),
#                     'content_map': content_map,
#                     'metadata': document_info
#                 }
#                 resource, created = Resource.objects.update_or_create(
#                     resource_id=document_id,
#                     defaults=resource_data
#                 )
                
#                 # Get document text
#                 document_text = doc_utils.extract_text_from_document(temp_file_path)
#                 formatted_text = doc_utils.format_processed_text(
#                     document_text, document_info, document_id
#                 ) if document_text and document_info else "Document text not available."
                
#                 response_data = {
#                     'message': 'Document processed',
#                     'resource': ResourceSerializer(resource).data,
#                     'summary_and_tags': summary_and_tags or "Summary and tags not available",
#                     'text_content': formatted_text
#                 }
                
#                 if not vector_store or not retrieval_chain:
#                     response_data['warning'] = 'Vector store or retrieval chain creation failed. Some features may be unavailable.'
                
#                 request.session[f'chat_history_{document_id}'] = chat_history
                
#                 return Response(response_data, status=status.HTTP_200_OK)
                
#             finally:
#                 # Clean up temporary file
#                 if os.path.exists(temp_file_path):
#                     os.remove(temp_file_path)
                    
#         except Exception as e:
#             logger.error(f"Error processing document: {str(e)}")
#             logger.error(traceback.format_exc())
#             return Response({'error': 'Failed to process document', 'details': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# views.py

class DocumentAnalysisView(APIView):
    def post(self, request):
        logger.info("=== Document Analysis Request ===")
        
        if 'document' not in request.FILES:
            return Response({
                'error': 'Document file is required'
            }, status=status.HTTP_400_BAD_REQUEST)

        document = request.FILES['document']
        
        try:
            # Generate document ID and create temp file
            document_id = str(uuid.uuid4())
            logger.info(f"Processing document ID: {document_id}")
            
            temp_file_path = handle_uploaded_file(document, document_id)
            logger.info(f"Temporary file created at: {temp_file_path}")
            
            try:
                # First get document info
                document_info = {
                    'title': document.name,
                    'file_type': document.content_type,
                    'file_size': document.size,
                    'modification_date': datetime.now().isoformat()
                }
                
                # Then process the document
                processing_result = doc_utils.process_document(document_id, temp_file_path)
                
                if not processing_result:
                    raise Exception("Document processing failed")
                
                vector_store, retrieval_chain, _, content_map, summary_and_tags, chat_history, processed_text = processing_result
                
                # Prepare resource data
                resource_data = {
                    'resource_id': document_id,
                    'resource_type': ResourceType.DOCUMENT,
                    'title': document_info['title'],
                    'content_map': content_map or {},
                    'metadata': {
                        **document_info,
                        'original_filename': document.name,
                        'file_size': document.size,
                        'content_type': document.content_type
                    }
                }
                
                # Create/update resource
                resource, created = Resource.objects.update_or_create(
                    resource_id=document_id,
                    defaults=resource_data
                )
                
                # Prepare response
                response_data = {
                    'message': 'Document processed successfully',
                    'resource': ResourceSerializer(resource).data,
                    'summary_and_tags': summary_and_tags or "Summary not available",
                    'text_content': processed_text or "Document text not available"
                }
                
                if not vector_store or not retrieval_chain:
                    response_data['warning'] = 'Vector store or retrieval chain creation failed'
                
                # Store chat history
                request.session[f'chat_history_{document_id}'] = chat_history
                
                return Response(response_data, status=status.HTTP_200_OK)
                
            finally:
                # Cleanup
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file removed: {temp_file_path}")
                    
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                'error': 'Failed to process document',
                'detail': str(e),
                'debug': {
                    'document_name': getattr(document, 'name', None),
                    'content_type': getattr(document, 'content_type', None),
                    'size': getattr(document, 'size', None)
                }
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)   

def handle_uploaded_file(file, document_id):
    """Helper function to handle file upload"""
    logger.info(f"Starting file upload for document_id: {document_id}")
    logger.info(f"Original filename: {file.name}")
    
    try:
        file_extension = os.path.splitext(file.name)[1]
        logger.info(f"File extension: {file_extension}")
        
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_documents', f"{document_id}{file_extension}")
        logger.info(f"Temp file path: {temp_path}")
        
        # Log directory creation
        directory = os.path.dirname(temp_path)
        logger.info(f"Creating directory if not exists: {directory}")
        os.makedirs(directory, exist_ok=True)
        
        # Log file size before writing
        logger.info(f"File size: {file.size} bytes")
        
        with open(temp_path, 'wb+') as destination:
            chunks_written = 0
            for chunk in file.chunks():
                destination.write(chunk)
                chunks_written += 1
            logger.info(f"Wrote {chunks_written} chunks to file")
        
        # Verify file was created
        if os.path.exists(temp_path):
            logger.info(f"File successfully saved at: {temp_path}")
            logger.info(f"Saved file size: {os.path.getsize(temp_path)} bytes")
        else:
            logger.error(f"File was not created at: {temp_path}")
            
        return temp_path
        
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        logger.error(traceback.format_exc())
        raise

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Resource
from .serializers import ResourceSerializer

class ResourceView(APIView):
    def post(self, request):
        """Create a new resource"""
        serializer = ResourceSerializer(data=request.data)
        if serializer.is_valid():
            resource = serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        """List all resources"""
        resources = Resource.objects.all()
        serializer = ResourceSerializer(resources, many=True)
        return Response(serializer.data)

class ResourceByIdView(APIView):
    def get(self, request, resource_id):
        """Get a resource by its resource_id field"""
        try:
            resource = Resource.objects.get(resource_id=resource_id)
            serializer = ResourceSerializer(resource)
            return Response(serializer.data)
        except Resource.DoesNotExist:
            return Response(
                {"error": "Resource not found"}, 
                status=status.HTTP_404_NOT_FOUND
            )
            

class TableOfContentsView(APIView):
    """API endpoint for extracting and retrieving document table of contents."""
    
    def get(self, request, resource_id):
        """Get table of contents for a specific document."""
        try:
            # Get the resource
            try:
                resource = Resource.objects.get(resource_id=resource_id)
            except Resource.DoesNotExist:
                return Response({
                    'error': 'Resource not found'
                }, status=status.HTTP_404_NOT_FOUND)

            # Check if content_map exists and has sections
            content_map = resource.content_map
            if not content_map or 'sections' not in content_map:
                return Response({
                    'error': 'Table of contents not available for this document'
                }, status=status.HTTP_404_NOT_FOUND)

            # Format TOC response
            toc_data = self._format_toc_response(content_map['sections'])

            return Response({
                'resource_id': resource_id,
                'title': resource.title,
                'toc': toc_data,
                'structure_type': content_map.get('structure_type', 'unknown')
            })

        except Exception as e:
            logger.error(f"Error retrieving TOC for resource {resource_id}: {str(e)}")
            return Response({
                'error': 'Failed to retrieve table of contents',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, resource_id):
        """Generate/update table of contents for a document."""
        try:
            # Get the resource
            try:
                resource = Resource.objects.get(resource_id=resource_id)
            except Resource.DoesNotExist:
                return Response({
                    'error': 'Resource not found'
                }, status=status.HTTP_404_NOT_FOUND)

            # Initialize document processor
            doc_processor = SmartDocumentProcessor()

            # Get the document text
            document_text = self._get_document_text(resource_id)
            if not document_text:
                return Response({
                    'error': 'Failed to retrieve document text'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Extract TOC using LLM
            sections = doc_processor.extract_toc_with_llm(document_text)
            
            # Update content map with new TOC
            content_map = resource.content_map or {}
            content_map['sections'] = [doc_processor._section_to_dict(section) for section in sections]
            content_map['structure_type'] = 'llm_generated'
            
            # Update resource
            resource.content_map = content_map
            resource.save()

            # Format TOC response
            toc_data = self._format_toc_response(content_map['sections'])

            return Response({
                'message': 'Table of contents generated successfully',
                'resource_id': resource_id,
                'title': resource.title,
                'toc': toc_data,
                'structure_type': content_map['structure_type']
            })

        except Exception as e:
            logger.error(f"Error generating TOC for resource {resource_id}: {str(e)}")
            return Response({
                'error': 'Failed to generate table of contents',
                'detail': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def _format_toc_response(self, sections: list) -> list:
        """Format sections into a hierarchical TOC structure."""
        formatted_toc = []
        
        for section in sections:
            section_data = {
                'title': section['title'],
                'level': section['level'],
                'chapter_num': section['chapter_num'],
                'page_range': section['page_range']
            }
            
            # Add subsections if they exist
            if section.get('subsections'):
                section_data['subsections'] = self._format_toc_response(section['subsections'])
            
            formatted_toc.append(section_data)
        
        return formatted_toc

    def _get_document_text(self, resource_id: str) -> str:
        """Retrieve document text from storage."""
        try:
            # Implementation depends on your storage setup
            # This is a placeholder - implement based on your storage system
            file_path = f"media/documents/{resource_id}"
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error retrieving document text: {str(e)}")
            return None