from django.apps import AppConfig

class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    
    def ready(self):
        """Initialize application resources when Django starts."""
        try:
            # Import here to avoid circular imports
            from .processors.nlp_resources import initialize_nlp_resources
            
            # Initialize NLP resources
            initialize_nlp_resources()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to initialize NLP resources: {str(e)}")
            # Don't raise the exception - let the app start even if NLP init fails
            # The document processor will handle the failure gracefully