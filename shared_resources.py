import logging
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class AIResources:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIResources, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("Initializing shared AI resources...")
        
        # Routing LLM (Fast)
        self.routing_llm = ChatOllama(
            model="gemma:2b", 
            base_url="http://localhost:11434", 
            temperature=0.2,
            num_predict=15,
            num_ctx=1024,
            keep_alive=-1
        )
        
        # Response LLM (Standard)
        self.response_llm = ChatOllama(
            model="gemma:2b",
            base_url="http://localhost:11434",
            temperature=0.2,
            num_predict=512,
            num_ctx=2048,
            keep_alive=-1
        )
        
        # Shared Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'} # Can change to 'cuda' if GPU available
        )
        
        self._initialized = True
        logger.info("Shared AI resources initialized")

# Global singleton instance
ai_resources = AIResources()
