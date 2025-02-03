import nltk
import logging
import spacy
from pathlib import Path

logger = logging.getLogger(__name__)

def initialize_nlp_resources():
    """Initialize all required NLP resources."""
    try:
        # Create nltk_data directory if it doesn't exist
        nltk_data_dir = Path.home() / 'nltk_data'
        nltk_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set NLTK data path
        nltk.data.path.append(str(nltk_data_dir))
        
        # Download required NLTK resources
        resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                logger.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, download_dir=str(nltk_data_dir))
        
        # Initialize spaCy
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model")
            return nlp
        except OSError:
            logger.info("Downloading spaCy model")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            return nlp
            
    except Exception as e:
        logger.error(f"Error initializing NLP resources: {str(e)}")
        raise