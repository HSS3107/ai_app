from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUserManager(BaseUserManager):
    """
    A custom manager for our User model.
    This is like a factory that knows how to create User objects.
    """
    def create_user(self, email, username, password=None, **extra_fields):
        """
        Create and return a regular user with an email, username and password.
        """
        if not email:
            raise ValueError('The Email field must be set')  # We insist on having an email!
        email = self.normalize_email(email)  # Normalize the email (lowercase the domain part)
        user = self.model(email=email, username=username, **extra_fields)
        user.set_password(password)  # Hash the password - very important for security!
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, password=None, **extra_fields):
        """
        Create and return a superuser - like a user, but with superpowers!
        """
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, username, password, **extra_fields)

class User(AbstractUser, PermissionsMixin):
    """
    Our custom User model.
    This is like defining a new type of user, tailored to our app's needs.
    """
    email = models.EmailField(unique=True)  # Every user must have a unique email
    username = models.CharField(max_length=30, unique=True)  # And a unique username
    is_active = models.BooleanField(default=True)  # Is this user active?
    is_staff = models.BooleanField(default=False)  # Is this user a staff member?
    date_joined = models.DateTimeField(auto_now_add=True)  # When did this user join?

    objects = CustomUserManager()  # Use our custom manager

    USERNAME_FIELD = 'email'  # We'll use email for logging in
    REQUIRED_FIELDS = ['username']  # We also require a username when creating a user

    def __str__(self):
        return self.email

# Add any other models your app needs here
# Models are like blueprints for your database tables

# models.py
from django.db import models
import uuid

# models.py
from django.db import models

from django.db import models

class ChatSession(models.Model):
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    session_identifier = models.CharField(max_length=255, blank=True, null=True)
    video_title = models.CharField(max_length=500, blank=True, null=True)  # Added
    video_id = models.CharField(max_length=50, blank=True, null=True)      # Added
    content_map = models.TextField(blank=True, null=True)                  # Added

    class Meta:
        db_table = 'app_chatsessions'

    def __str__(self):
        return f"Chat {self.id} - {self.created_at}"