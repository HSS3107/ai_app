from rest_framework import serializers
from .models import ChatSession, WaitList, User

class UserSerializer(serializers.ModelSerializer):
    """
    Serializer for our User model.
    Serializers are like translators between Python objects and JSON.
    """
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'password')
        extra_kwargs = {'password': {'write_only': True}}  # Password is write-only for security

    def create(self, validated_data):
        """
        Create and return a new user, given the validated data.
        This is like a factory method for creating users.
        """
        user = User.objects.create_user(**validated_data)
        return user

# Add any other serializers your app needs here

# Serializers help convert complex data types (like Django models) to Python datatypes 
# that can then be easily rendered into JSON

class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = ['id', 'question', 'answer', 'created_at', 'session_identifier', 
                 'video_title', 'video_id', 'content_map']
        read_only_fields = ['id', 'created_at']

class WaitListSerializer(serializers.ModelSerializer):
    class Meta:
        model = WaitList
        fields = ['id', 'created_at', 'name', 'email', 'phone_no']
        read_only_fields = ['id', 'created_at']
