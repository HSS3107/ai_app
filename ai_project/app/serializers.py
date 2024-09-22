from rest_framework import serializers
from .models import User

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
# Serializers help convert complex data types (like Django models) to Python datatypes that can then be easily rendered into JSON