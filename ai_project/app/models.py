from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUserManager(BaseUserManager):
    def create_user(self, email, username, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, username=username, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, username, password, **extra_fields)

class User(AbstractUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=30, unique=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    date_joined = models.DateTimeField(auto_now_add=True)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email

class ResourceType(models.TextChoices):
    VIDEO = 'VIDEO', 'Video'
    DOCUMENT = 'DOCUMENT', 'Document'

class Resource(models.Model):
    """
    Base model for all processable resources (videos, documents)
    """
    resource_id = models.CharField(max_length=255, unique=True)
    resource_type = models.CharField(
        max_length=20,
        choices=ResourceType.choices,
        default=ResourceType.VIDEO
    )
    title = models.CharField(max_length=500)
    content_map = models.TextField(blank=True, null=True)
    metadata = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resources'
    )

    class Meta:
        db_table = 'app_resources'
        indexes = [
            models.Index(fields=['resource_type', 'resource_id']),
            models.Index(fields=['user', 'created_at']),
        ]

    def __str__(self):
        return f"{self.resource_type} - {self.title}"

class VideoMetadata(models.Model):
    """
    Additional metadata specific to videos
    """
    resource = models.OneToOneField(
        Resource,
        on_delete=models.CASCADE,
        related_name='video_metadata'
    )
    channel_name = models.CharField(max_length=255)
    channel_id = models.CharField(max_length=255)
    duration_seconds = models.IntegerField()
    view_count = models.BigIntegerField(null=True, blank=True)
    like_count = models.BigIntegerField(null=True, blank=True)
    published_at = models.DateTimeField()
    captions_type = models.CharField(max_length=50)
    language = models.CharField(max_length=10, default='en')
    thumbnail_url = models.URLField(max_length=500, blank=True, null=True)

    class Meta:
        db_table = 'app_video_metadata'
        indexes = [
            models.Index(fields=['channel_id']),
            models.Index(fields=['published_at']),
        ]

    def __str__(self):
        return f"Video Metadata - {self.resource.title}"

class DocumentMetadata(models.Model):
    """
    Additional metadata specific to documents
    """
    resource = models.OneToOneField(
        Resource,
        on_delete=models.CASCADE,
        related_name='document_metadata'
    )
    file_type = models.CharField(max_length=50)
    file_size = models.BigIntegerField()
    page_count = models.IntegerField(null=True, blank=True)
    author = models.CharField(max_length=255, blank=True, null=True)
    publication_date = models.DateField(null=True, blank=True)
    last_modified = models.DateTimeField()

    class Meta:
        db_table = 'app_document_metadata'
        indexes = [
            models.Index(fields=['file_type']),
            models.Index(fields=['last_modified']),
        ]

    def __str__(self):
        return f"Document Metadata - {self.resource.title}"

class ChatSession(models.Model):
    """
    Enhanced chat session model that can handle both videos and documents
    """
    resource = models.ForeignKey(
        Resource,
        null=True,
        on_delete=models.CASCADE,
        related_name='chat_sessions'
    )
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    session_identifier = models.CharField(max_length=255, blank=True, null=True)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='chat_sessions'
    )
    processing_time = models.FloatField(null=True, blank=True)
    tokens_used = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = 'app_chatsessions'
        indexes = [
            models.Index(fields=['session_identifier']),
            models.Index(fields=['created_at']),
            models.Index(fields=['user', 'created_at']),
        ]

    def __str__(self):
        return f"Chat {self.id} - {self.resource.title} - {self.created_at}"

class WaitList(models.Model):
    """
    WaitList model for storing potential user information
    """
    name = models.TextField()
    email = models.EmailField(max_length=254, unique=False)
    phone_no = models.CharField(max_length=15, unique=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'app_waitlist'

    def __str__(self):
        return f"Waitlist Entry - {self.name} - {self.created_at}"