from rest_framework import serializers
from .models import (
    User, Resource, VideoMetadata, DocumentMetadata, 
    ChatSession, WaitList
)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password', 'date_joined')
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user

class VideoMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoMetadata
        fields = [
            'channel_name', 'channel_id', 'duration_seconds',
            'view_count', 'like_count', 'published_at',
            'captions_type', 'language', 'thumbnail_url'
        ]

class DocumentMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentMetadata
        fields = [
            'file_type', 'file_size', 'page_count',
            'author', 'publication_date', 'last_modified'
        ]

class ResourceSerializer(serializers.ModelSerializer):
    video_metadata = VideoMetadataSerializer(read_only=True)
    document_metadata = DocumentMetadataSerializer(read_only=True)
    
    class Meta:
        model = Resource
        fields = [
            'id', 'resource_id', 'resource_type', 'title',
            'content_map', 'metadata', 'created_at', 'updated_at',
            'video_metadata', 'document_metadata'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']

    def to_representation(self, instance):
        """
        Customize the output based on resource type
        """
        representation = super().to_representation(instance)
        
        # Remove irrelevant metadata based on resource type
        if instance.resource_type == 'VIDEO':
            representation.pop('document_metadata', None)
        else:
            representation.pop('video_metadata', None)
            
        return representation

class ChatSessionSerializer(serializers.ModelSerializer):
    resource = ResourceSerializer(read_only=True)
    resource_id = serializers.CharField(write_only=True, required=True)
    
    class Meta:
        model = ChatSession
        fields = [
            'id', 'resource', 'resource_id', 'question', 'answer',
            'created_at', 'session_identifier', 'processing_time',
            'tokens_used'
        ]
        read_only_fields = ['id', 'created_at', 'processing_time', 'tokens_used']

    def create(self, validated_data):
        resource_id = validated_data.pop('resource_id')
        try:
            resource = Resource.objects.get(resource_id=resource_id)
            return ChatSession.objects.create(resource=resource, **validated_data)
        except Resource.DoesNotExist:
            raise serializers.ValidationError({'resource_id': 'Resource not found'})

class WaitListSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaitList
        fields = ['id', 'created_at', 'name', 'email', 'phone_no']
        read_only_fields = ['id', 'created_at']

# Specialized serializers for specific use cases

class ResourceListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for listing resources
    """
    resource_type_display = serializers.CharField(source='get_resource_type_display')
    
    class Meta:
        model = Resource
        fields = ['id', 'resource_id', 'title', 'resource_type', 
                 'resource_type_display', 'created_at']

class ChatSessionListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for listing chat sessions
    """
    resource_title = serializers.CharField(source='resource.title')
    resource_type = serializers.CharField(source='resource.resource_type')
    
    class Meta:
        model = ChatSession
        fields = ['id', 'resource_title', 'resource_type', 'question', 
                 'created_at', 'session_identifier']

class ResourceDetailSerializer(ResourceSerializer):
    """
    Extended resource serializer with chat history
    """
    chat_sessions = ChatSessionListSerializer(many=True, read_only=True)
    
    class Meta(ResourceSerializer.Meta):
        fields = ResourceSerializer.Meta.fields + ['chat_sessions']
        
class ResourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resource
        fields = ['id', 'resource_id', 'resource_type', 'title', 'content_map', 'metadata', 'created_at', 'updated_at']
        read_only_fields = ['created_at', 'updated_at']