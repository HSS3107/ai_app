import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
import logging

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types."""
    def default(self, obj):
        if isinstance(obj, Document):
            return {
                "page_content": obj.page_content,
                "metadata": obj.metadata,
                "_type": "Document"
            }
        if isinstance(obj, BaseMessage):
            return {
                "content": obj.content,
                "type": obj.type,
                "_type": "BaseMessage"
            }
        if isinstance(obj, datetime):
            return obj.isoformat()
        try:
            # Try to get the dict representation of the object
            return obj.__dict__
        except AttributeError:
            try:
                # Try to convert to string if dict representation not available
                return str(obj)
            except Exception:
                return f"<Non-serializable object of type {type(obj).__name__}>"

class ChainLogger:
    def __init__(self, base_dir: str = "Chain"):
        """
        Initialize the ChainLogger.
        
        Args:
            base_dir (str): Base directory for storing chain execution logs
        """
        self.base_dir = base_dir
        self._ensure_directory_exists()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger if it doesn't already have handlers
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            
        # Create file handler
        log_file_path = os.path.join(self.base_dir, 'chain_logger.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        
        # Add file handler if not already present
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)
    
    def _ensure_directory_exists(self):
        """Create the base directory if it doesn't exist."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            self.logger.info(f"Created directory: {self.base_dir}")
    
    def _serialize_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        Serialize BaseMessage objects to dictionaries.
        
        Args:
            messages (List[BaseMessage]): List of chat messages
            
        Returns:
            List[Dict[str, str]]: Serialized messages
        """
        return [{"role": msg.type, "content": msg.content} for msg in messages]
    
    def _serialize_context(self, context: Any) -> Any:
        """
        Serialize context data, handling Document objects specifically.
        
        Args:
            context: Context data which might contain Document objects
            
        Returns:
            Serialized context data
        """
        if isinstance(context, Document):
            return {
                "page_content": context.page_content,
                "metadata": context.metadata,
                "_type": "Document"
            }
        elif isinstance(context, list):
            return [self._serialize_context(item) for item in context]
        elif isinstance(context, dict):
            return {k: self._serialize_context(v) for k, v in context.items()}
        return context
    
    def _get_log_filename(self, query_id: Optional[str] = None) -> str:
        """
        Generate a filename for the chain execution log.
        
        Args:
            query_id (Optional[str]): Unique identifier for the query
            
        Returns:
            str: Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if query_id:
            return f"chain_execution_{query_id}_{timestamp}.json"
        return f"chain_execution_{timestamp}.json"
    
    def log_chain_execution(self, 
                          query: str,
                          chat_history: List[BaseMessage],
                          context: Any,
                          response: str,
                          document_title: str,
                          content_map: str,
                          retriever_info: Dict[str, Any],
                          query_id: Optional[str] = None) -> str:
        """
        Log a chain execution with all relevant details.
        
        Args:
            query (str): User's query
            chat_history (List[BaseMessage]): Conversation history
            context (Any): Retrieved context chunks
            response (str): Generated response
            document_title (str): Title of the document being processed
            content_map (str): Document's content map
            retriever_info (Dict[str, Any]): Information about the retrieval process
            query_id (Optional[str]): Unique identifier for the query
            
        Returns:
            str: Path to the created log file
        """
        try:
            # Create the execution log
            execution_log = {
                "timestamp": datetime.now().isoformat(),
                "document_title": document_title,
                "query": query,
                "chat_history": self._serialize_messages(chat_history),
                "retrieval_details": {
                    "context_chunks": self._serialize_context(context),
                    "retriever_info": retriever_info
                },
                "content_map": content_map,
                "response": response,
                "metadata": {
                    "query_id": query_id,
                    "execution_time": datetime.now().isoformat()
                }
            }
            
            # Generate filename and full path
            filename = self._get_log_filename(query_id)
            file_path = os.path.join(self.base_dir, filename)
            
            # Write the log file using the custom encoder
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(execution_log, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
            
            self.logger.info(f"Chain execution log saved to: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error logging chain execution: {str(e)}")
            raise
    
    def get_latest_execution(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent chain execution log.
        
        Returns:
            Optional[Dict[str, Any]]: Most recent execution log or None if no logs exist
        """
        try:
            # Get all log files
            log_files = [f for f in os.listdir(self.base_dir) if f.startswith('chain_execution_')]
            
            if not log_files:
                return None
            
            # Sort by timestamp and get the latest
            latest_file = max(log_files, key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)))
            file_path = os.path.join(self.base_dir, latest_file)
            
            # Read and return the latest log
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error retrieving latest execution: {str(e)}")
            return None
    
    def clear_old_logs(self, days_to_keep: int = 7):
        """
        Remove log files older than the specified number of days.
        
        Args:
            days_to_keep (int): Number of days to keep logs for
        """
        try:
            current_time = datetime.now().timestamp()
            for filename in os.listdir(self.base_dir):
                file_path = os.path.join(self.base_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getctime(file_path)
                    if file_age > (days_to_keep * 24 * 60 * 60):
                        os.remove(file_path)
                        self.logger.info(f"Removed old log file: {filename}")
                        
        except Exception as e:
            self.logger.error(f"Error clearing old logs: {str(e)}")