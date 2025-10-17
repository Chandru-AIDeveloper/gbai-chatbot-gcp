import json
import os
import logging
import traceback
import re
import uuid
import asyncio
import faiss
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Google Cloud - Async libraries
from google.cloud import firestore
from google.cloud import storage

# Bot Imports
try:
    import formula_bot
    FORMULA_BOT_AVAILABLE = True
except ImportError:
    FORMULA_BOT_AVAILABLE = False
    logging.warning("Formula bot not available")

try:
    import report_bot
    REPORT_BOT_AVAILABLE = True
except ImportError:
    REPORT_BOT_AVAILABLE = False
    logging.warning("Report bot not available")

try:
    import menu_bot
    MENU_BOT_AVAILABLE = True
except ImportError:
    MENU_BOT_AVAILABLE = False
    logging.warning("Menu bot not available")

try:
    import project_bot
    PROJECT_BOT_AVAILABLE = True
except ImportError:
    PROJECT_BOT_AVAILABLE = False
    logging.warning("Project bot not available")

try:
    import general_bot
    GENERAL_BOT_AVAILABLE = True
except ImportError:
    GENERAL_BOT_AVAILABLE = False
    logging.warning("General bot not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# CONFIGURATION FROM ENVIRONMENT
# ===========================
class Config:
    """Centralized configuration from environment variables."""
    # GCS Configuration
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "ai-chatbot-test-473604-memory-store")
    VECTORSTORE_INDEX_PATH = os.getenv("VECTORSTORE_INDEX_PATH", "vectorstore/faiss.index")
    VECTORSTORE_PKL_PATH = os.getenv("VECTORSTORE_PKL_PATH", "vectorstore/faiss.pkl")
    
    # Ollama Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma:2b")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Performance Settings
    TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false")
    OMP_NUM_THREADS = os.getenv("OMP_NUM_THREADS", "2")
    
    # Memory Settings
    MEMORY_SAVE_FREQUENCY = int(os.getenv("MEMORY_SAVE_FREQUENCY", "5"))
    MEMORY_RETRIEVAL_COUNT = int(os.getenv("MEMORY_RETRIEVAL_COUNT", "3"))
    
    # Thread Settings
    MAX_THREADS_PER_USER = int(os.getenv("MAX_THREADS_PER_USER", "50"))
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "90"))
    
    # Server Settings
    PORT = int(os.getenv("PORT", "8080"))
    
    @property
    def ollama_base_url(self):
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

config = Config()

# Performance optimizations
os.environ["TOKENIZERS_PARALLELISM"] = config.TOKENIZERS_PARALLELISM
os.environ["OMP_NUM_THREADS"] = config.OMP_NUM_THREADS

# ===========================
# GOOGLE CLOUD STORAGE SETUP
# ===========================
db = firestore.AsyncClient()
storage_client = storage.Client()
bucket = storage_client.bucket(config.GCS_BUCKET_NAME)

# ===========================
# GCS HELPER FUNCTIONS
# ===========================
def gcs_read_json(gcs_path: str) -> dict:
    """Reads a JSON file from GCS."""
    try:
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            logger.warning(f"GCS path {gcs_path} does not exist")
            return {}
        return json.loads(blob.download_as_string())
    except Exception as e:
        logger.error(f"Failed to read JSON from GCS path {gcs_path}: {e}")
        return {}

def gcs_write_json(gcs_path: str, data: dict):
    """Writes a dictionary to a JSON file in GCS."""
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(json.dumps(data, indent=4), content_type='application/json')
        logger.info(f"Successfully wrote JSON to {gcs_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON to GCS path {gcs_path}: {e}")

def gcs_read_bytes(gcs_path: str) -> bytes:
    """Reads bytes from GCS."""
    try:
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            raise FileNotFoundError(f"GCS path {gcs_path} does not exist")
        return blob.download_as_bytes()
    except Exception as e:
        logger.error(f"Failed to read bytes from GCS path {gcs_path}: {e}")
        raise

def gcs_write_bytes(gcs_path: str, data: bytes):
    """Writes bytes to GCS."""
    try:
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(data)
        logger.info(f"Successfully wrote bytes to {gcs_path}")
    except Exception as e:
        logger.error(f"Failed to write bytes to GCS path {gcs_path}: {e}")
        raise

def gcs_read_text(gcs_path: str) -> str:
    """Reads a text file from GCS."""
    try:
        blob = bucket.blob(gcs_path)
        if not blob.exists():
            logger.warning(f"GCS path {gcs_path} does not exist")
            return ""
        return blob.download_as_string().decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to read text from GCS path {gcs_path}: {e}")
        return ""

def gcs_list_files(prefix: str) -> List[str]:
    """Lists all files in a GCS 'folder'."""
    try:
        blobs = storage_client.list_blobs(config.GCS_BUCKET_NAME, prefix=prefix)
        return [blob.name for blob in blobs]
    except Exception as e:
        logger.error(f"Failed to list files with prefix {prefix}: {e}")
        return []

def gcs_blob_exists(gcs_path: str) -> bool:
    """Check if a blob exists in GCS."""
    return bucket.blob(gcs_path).exists()

# ===========================
# LOGIN DTO
# ===========================
class LoginDTO(BaseModel):
    """Data Transfer Object for login information."""
    UserName: str = "anonymous"

def get_login_details(Login: str = Header(...)) -> LoginDTO:
    """Parse and validate login header."""
    try:
        return LoginDTO(**json.loads(Login))
    except Exception as e:
        logger.error(f"Invalid login header: {e}")
        raise HTTPException(status_code=400, detail="Invalid login header")

# ===========================
# GREETING HANDLER (HARDCODED)
# ===========================
class GreetingHandler:
    """Handles greeting messages with predefined responses."""
    GREETINGS = {
        "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "hola", "namaste", "vanakkam"
    }
    
    RESPONSES = [
        "Hello! I'm the GoodBooks Technologies AI assistant. How can I help you with our ERP system today?",
        "Hi there! Welcome to GoodBooks Technologies. What would you like to know about our ERP modules?",
        "Greetings! I'm here to assist you with GoodBooks ERP. What can I help you with?",
        "Hello! Ready to help you explore GoodBooks Technologies' solutions. What's on your mind?"
    ]
    
    @classmethod
    def is_greeting(cls, message: str) -> bool:
        """Check if message is a greeting."""
        msg_lower = message.lower().strip()
        # Check for exact match or greeting at start
        return (msg_lower in cls.GREETINGS or 
                any(msg_lower.startswith(greet) for greet in cls.GREETINGS))
    
    @classmethod
    def get_response(cls) -> str:
        """Get a random greeting response."""
        import random
        return random.choice(cls.RESPONSES)

# ===========================
# GOODBOOKS CONTENT VALIDATOR
# ===========================
class GoodBooksValidator:
    """Validates that queries and responses are GoodBooks-related."""
    
    GOODBOOKS_KEYWORDS = {
        # Company and product
        "goodbooks", "erp", "technologies",
        
        # Modules
        "inventory", "hrms", "payroll", "customer", "security", "module",
        "accounting", "finance", "sales", "purchase", "manufacturing",
        
        # Features
        "employee", "salary", "report", "dashboard", "analysis",
        "formula", "calculation", "menu", "navigation", "project",
        "user", "role", "permission", "workflow", "approval",
        
        # Business terms
        "company", "organization", "policy", "procedure", "compliance",
        "audit", "transaction", "record", "data", "system"
    }
    
    OUT_OF_SCOPE_PATTERNS = [
        r"\b(weather|sports|news|politics|celebrity|entertainment)\b",
        r"\b(recipe|cooking|food|restaurant)\b",
        r"\b(movie|film|tv show|series)\b",
        r"\b(game|gaming|video game)\b",
        r"\b(travel|vacation|tourism|hotel)\b",
        r"\b(medical|health|disease|symptom)\b"
    ]
    
    @classmethod
    def is_goodbooks_related(cls, text: str, context: str = "") -> bool:
        """Check if query is related to GoodBooks."""
        combined = f"{text} {context}".lower()
        
        # Check for GoodBooks keywords
        has_keywords = any(keyword in combined for keyword in cls.GOODBOOKS_KEYWORDS)
        
        # Check for out-of-scope patterns
        is_out_of_scope = any(re.search(pattern, combined, re.IGNORECASE) 
                              for pattern in cls.OUT_OF_SCOPE_PATTERNS)
        
        return has_keywords and not is_out_of_scope
    
    @classmethod
    def get_out_of_scope_response(cls) -> str:
        """Response for out-of-scope queries."""
        return ("I'm a specialized assistant for GoodBooks Technologies ERP system. "
                "I can only help with questions about GoodBooks modules, features, "
                "company information, and related ERP functionalities. "
                "Please ask me something about GoodBooks ERP.")

# ===========================
# ENHANCED MEMORY SYSTEM WITH GCS
# ===========================
class EnhancedConversationalMemory:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None
        self.is_loaded = False
        self.memories_collection = db.collection("conversational_memories")
        self.memory_counter = 0
        
        self.memory_categories = {
            "conversation": "General conversation turn",
            "context": "Important context or preference",
            "task": "Task completion or work-related",
            "personal": "Personal information or preference",
            "error": "Error or issue resolution",
            "query_detail": "Detailed question about previous topic",
            "format_request": "Request for different format/presentation",
            "greeting": "User greeting or casual chat"
        }

    def _blocking_load_from_gcs(self) -> FAISS:
        """Loads the FAISS index from GCS (separate index and pkl files)."""
        index_blob = bucket.blob(config.VECTORSTORE_INDEX_PATH)
        pkl_blob = bucket.blob(config.VECTORSTORE_PKL_PATH)
        
        if not (index_blob.exists() and pkl_blob.exists()):
            raise FileNotFoundError("Vector store files not found in Cloud Storage.")
        
        # Download files
        index_bytes = gcs_read_bytes(config.VECTORSTORE_INDEX_PATH)
        docstore_bytes = gcs_read_bytes(config.VECTORSTORE_PKL_PATH)
        
        # Deserialize
        index = faiss.deserialize_index(index_bytes)
        docstore = pickle.loads(docstore_bytes)
        
        # Reconstruct FAISS
        return FAISS(
            embedding_function=self.embeddings.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=getattr(docstore, 'index_to_docstore_id', {})
        )

    def _blocking_save_to_gcs(self):
        """Saves the FAISS index to GCS (separate index and pkl files)."""
        if not self.vectorstore:
            return
        
        # Serialize to bytes
        index_bytes = faiss.serialize_index(faiss.clone_index(self.vectorstore.index))
        docstore_bytes = pickle.dumps(self.vectorstore.docstore)
        
        # Upload to GCS
        gcs_write_bytes(config.VECTORSTORE_INDEX_PATH, index_bytes)
        gcs_write_bytes(config.VECTORSTORE_PKL_PATH, docstore_bytes)
        
        logger.info(f"Vector store saved to GCS")

    async def load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one."""
        if self.is_loaded:
            return
        
        try:
            logger.info("Attempting to load vector store from Cloud Storage...")
            self.vectorstore = await asyncio.to_thread(self._blocking_load_from_gcs)
            logger.info("Successfully loaded vector store.")
        except FileNotFoundError:
            logger.warning("Vector store not found. Creating a new one.")
            dummy_doc = Document(
                page_content="GoodBooks Technologies ERP system initialization",
                metadata={"username": "system", "category": "system"}
            )
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            await self.save_vectorstore()
        except Exception as e:
            logger.error(f"Unexpected error loading vector store: {traceback.format_exc()}")
            dummy_doc = Document(
                page_content="GoodBooks Technologies ERP system initialization",
                metadata={"username": "system", "category": "system"}
            )
            self.vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
        
        self.is_loaded = True

    async def save_vectorstore(self):
        """Save vectorstore to GCS."""
        if not self.vectorstore:
            return
        
        logger.info("Saving vector store to Cloud Storage...")
        try:
            await asyncio.to_thread(self._blocking_save_to_gcs)
            logger.info("Vector store saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save vector store: {traceback.format_exc()}")

    def categorize_interaction(self, user_message: str, bot_response: str, bot_type: str) -> str:
        """Categorize interaction type."""
        user_lower = user_message.lower()
        
        # Check for greeting
        if GreetingHandler.is_greeting(user_message):
            return "greeting"
        
        # Check for format requests
        if any(phrase in user_lower for phrase in ["table format", "in table", "tabular", "show in table", "detailed", "elaborate"]):
            return "format_request"
        
        # Check for follow-up questions
        if any(phrase in user_lower for phrase in ["about that", "more about", "detail about", "explain that"]):
            return "query_detail"
        
        # Check for personal info
        if any(word in user_lower for word in ["my name is", "i am", "i like", "i prefer"]):
            return "personal"
        
        # Check for task-related
        if bot_type in ["formula", "report", "project"] or any(word in user_lower for word in ["calculate", "analyze"]):
            return "task"
        
        return "conversation"

    def extract_simple_topic(self, user_message: str, bot_response: str) -> str:
        """Extract topic from message."""
        combined_text = f"{user_message}".lower()
        
        topic_keywords = {
            "modules": ["module", "goodbooks", "system", "erp"],
            "payroll": ["payroll", "salary", "wage", "compensation"],
            "inventory": ["inventory", "stock", "warehouse", "item"],
            "security": ["security", "permission", "role", "access"],
            "calculations": ["calculate", "formula", "compute", "math"],
            "reports": ["report", "data", "analysis", "dashboard"],
            "projects": ["project", "task", "milestone"],
            "hrms": ["employee", "hr", "attendance", "leave"],
            "customer": ["customer", "client", "crm"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return topic
        
        return "general"

    async def add_memory(self, username: str, user_message: str, bot_response: str, bot_type: str, thread_id: str):
        """Store conversation turn in both vectorstore and Firestore."""
        await self.load_or_create_vectorstore()
        
        timestamp = datetime.now().isoformat()
        memory_id = f"{username}_{self.memory_counter}_{int(datetime.now().timestamp())}"
        category = self.categorize_interaction(user_message, bot_response, bot_type)
        topic = self.extract_simple_topic(user_message, bot_response)
        
        # Create memory document
        conversation_context = f"User: {user_message} | Bot ({bot_type}): {bot_response[:200]}..."
        memory_doc = Document(
            page_content=conversation_context,
            metadata={
                "memory_id": memory_id,
                "username": username,
                "timestamp": timestamp,
                "user_message": user_message,
                "bot_response": bot_response[:500],
                "bot_type": bot_type,
                "category": category,
                "topic": topic,
                "thread_id": thread_id
            }
        )
        
        # Add to vectorstore
        if self.vectorstore:
            self.vectorstore.add_documents([memory_doc])
        
        # Store in Firestore
        doc_data = {
            "memory_id": memory_id,
            "username": username,
            "timestamp": timestamp,
            "user_message": user_message,
            "bot_response": bot_response[:500],
            "bot_type": bot_type,
            "category": category,
            "topic": topic,
            "thread_id": thread_id
        }
        await self.memories_collection.document(memory_id).set(doc_data)
        
        self.memory_counter += 1
        
        # Batch save based on config
        if self.memory_counter % config.MEMORY_SAVE_FREQUENCY == 0:
            asyncio.create_task(self.save_vectorstore())
        
        logger.info(f"Stored memory {memory_id} for {username} in thread {thread_id}")

    async def retrieve_memories(self, username: str, query: str, k: int = None, thread_id: str = None, thread_isolation: bool = False) -> List[Dict]:
        """Retrieve relevant memories with thread filtering."""
        if k is None:
            k = config.MEMORY_RETRIEVAL_COUNT
        
        await self.load_or_create_vectorstore()
        
        if not self.vectorstore:
            return []
        
        try:
            # Enhanced queries
            enhanced_queries = [query, f"User {username}: {query}"]
            
            all_relevant_docs = []
            for enhanced_query in enhanced_queries[:2]:
                docs = self.vectorstore.similarity_search(enhanced_query, k=k * 3)
                all_relevant_docs.extend(docs)
            
            # Filter and score
            user_memories = {}
            for doc in all_relevant_docs:
                if doc.metadata.get("username") == username and doc.metadata.get("memory_id") != "system":
                    # Thread isolation
                    if thread_isolation and thread_id:
                        if doc.metadata.get("thread_id") != thread_id:
                            continue
                    
                    memory_id = doc.metadata.get("memory_id")
                    if memory_id not in user_memories:
                        relevance_score = self.calculate_relevance_score(doc, query)
                        
                        # Boost same thread
                        if not thread_isolation and thread_id and doc.metadata.get("thread_id") == thread_id:
                            relevance_score += 0.5
                        
                        user_memories[memory_id] = {
                            "memory_id": memory_id,
                            "timestamp": doc.metadata.get("timestamp"),
                            "user_message": doc.metadata.get("user_message"),
                            "bot_response": doc.metadata.get("bot_response"),
                            "bot_type": doc.metadata.get("bot_type"),
                            "category": doc.metadata.get("category"),
                            "topic": doc.metadata.get("topic"),
                            "thread_id": doc.metadata.get("thread_id"),
                            "relevance_score": relevance_score,
                            "content": doc.page_content
                        }
            
            sorted_memories = sorted(user_memories.values(), 
                                   key=lambda x: (x["relevance_score"], x["timestamp"]), 
                                   reverse=True)
            result = sorted_memories[:k]
            
            logger.info(f"Retrieved {len(result)} memories for {username}")
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def calculate_relevance_score(self, doc: Document, current_query: str) -> float:
        """Calculate relevance score based on recency and content match."""
        score = 0.0
        
        # Time-based scoring
        try:
            doc_time = datetime.fromisoformat(doc.metadata.get("timestamp"))
            time_diff = datetime.now() - doc_time
            
            if time_diff.days == 0:
                score = 1.0
            elif time_diff.days <= 7:
                score = 0.8
            elif time_diff.days <= 30:
                score = 0.5
            else:
                score = 0.3
        except:
            score = 0.1
        
        # Keyword matching
        current_lower = current_query.lower()
        doc_content_lower = doc.page_content.lower()
        common_words = len(set(current_lower.split()) & set(doc_content_lower.split()))
        score += common_words * 0.1
        
        return score

# ===========================
# CONVERSATION THREAD MANAGER
# ===========================
class ConversationHistoryManager:
    def __init__(self):
        self.threads_ref = db.collection("conversation_threads")

    async def create_new_thread(self, username: str, initial_message: str = None) -> str:
        """Create new conversation thread."""
        thread_id = str(uuid.uuid4())
        title = self._generate_title(initial_message) if initial_message else "New Conversation"
        
        doc = {
            "thread_id": thread_id,
            "username": username,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "is_active": True,
            "message_count": 0
        }
        
        await self.threads_ref.document(thread_id).set(doc)
        logger.info(f"Created thread {thread_id} for {username}")
        return thread_id

    def _generate_title(self, message: str) -> str:
        """Generate title from message."""
        if not message:
            return "New Conversation"
        
        title = re.sub(r'^(what is|tell me about|how to|can you)\s+', '', 
                      message.strip(), flags=re.IGNORECASE)
        return (title[:47] + "...") if len(title) > 50 else (title.capitalize() or "New Conversation")

    async def add_message_to_thread(self, thread_id: str, user_message: str, bot_response: str, bot_type: str):
        """Add message to thread with atomic update."""
        thread_ref = self.threads_ref.document(thread_id)
        message = {
            "id": str(uuid.uuid4()),
            "user_message": user_message,
            "bot_response": bot_response,
            "bot_type": bot_type,
            "timestamp": datetime.now().isoformat()
        }
        
        await thread_ref.update({
            "messages": firestore.ArrayUnion([message]),
            "updated_at": datetime.now().isoformat(),
            "message_count": firestore.Increment(1)
        })

    async def get_thread(self, thread_id: str):
        """Get thread by ID."""
        doc = await self.threads_ref.document(thread_id).get()
        return doc.to_dict() if doc.exists else None

    async def get_user_threads(self, username: str, limit: int = None):
        """Get all threads for user."""
        if limit is None:
            limit = config.MAX_THREADS_PER_USER
        
        query = (self.threads_ref
                .where("username", "==", username)
                .where("is_active", "==", True)
                .order_by("updated_at", direction=firestore.Query.DESCENDING)
                .limit(limit))
        
        docs_stream = query.stream()
        return [doc.to_dict() async for doc in docs_stream]

    async def delete_thread(self, thread_id: str, username: str) -> bool:
        """Soft delete thread."""
        thread_ref = self.threads_ref.document(thread_id)
        doc = await thread_ref.get()
        
        if doc.exists and doc.to_dict().get("username") == username:
            await thread_ref.update({"is_active": False})
            return True
        return False

    async def rename_thread(self, thread_id: str, username: str, new_title: str) -> bool:
        """Rename thread."""
        thread_ref = self.threads_ref.document(thread_id)
        doc = await thread_ref.get()
        
        if doc.exists and doc.to_dict().get("username") == username:
            await thread_ref.update({
                "title": new_title,
                "updated_at": datetime.now().isoformat()
            })
            return True
        return False

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'batch_size': 1}
)
enhanced_memory = EnhancedConversationalMemory(embeddings)
history_manager = ConversationHistoryManager()

# ===========================
# BASE BOT WRAPPER (REUSABLE)
# ===========================
class BaseBotWrapper:
    """Base wrapper for all bot types to reduce code duplication."""
    
    def __init__(self, bot_module, bot_name: str, is_available: bool, chat_method: str = "chat"):
        self.bot_module = bot_module
        self.bot_name = bot_name
        self.is_available = is_available
        self.chat_method = chat_method

    async def answer(self, question: str, context: str = "", format_request: str = None) -> str:
        """Generic answer method for all bots."""
        if not self.is_available:
            return f"{self.bot_name.capitalize()} service is currently unavailable."
        
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            message = MockMessage(question)
            
            # Get the appropriate chat method from the bot module
            chat_func = getattr(self.bot_module, self.chat_method)
            result = await chat_func(message, Login='{"UserName": "orchestrator"}')
            
            # Handle JSONResponse objects
            if isinstance(result, JSONResponse):
                try:
                    body_content = json.loads(
                        result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                    )
                    return body_content.get("response", "An unknown error occurred.")
                except Exception as e:
                    logger.error(f"Could not parse JSONResponse body: {e}")
                    return "An unreadable error response was received from the bot."
            
            # Handle dict responses
            if isinstance(result, dict):
                return result.get("response", str(result))
            
            return str(result)
            
        except Exception as e:
            logger.error(f"{self.bot_name} bot error: {traceback.format_exc()}")
            return f"I encountered an issue with {self.bot_name} service."

# ===========================
# GENERAL BOT WRAPPER (SPECIAL HANDLING)
# ===========================
class GeneralBotWrapper:
    """Special wrapper for general bot with enhanced validation."""
    
    @staticmethod
    async def answer(question: str, context: str, format_request: str = None) -> str:
        if not GENERAL_BOT_AVAILABLE:
            return "General information service is currently unavailable."
        
        try:
            # Validate GoodBooks context
            if not GoodBooksValidator.is_goodbooks_related(question, context):
                return GoodBooksValidator.get_out_of_scope_response()
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            enhanced_question = GeneralBotWrapper.enhance_question_with_context(
                question, context, format_request
            )
            message = MockMessage(enhanced_question)
            login_header = '{"UserName": "orchestrator"}'
            
            result = await general_bot.chat(message, Login=login_header)
            
            # Handle JSONResponse objects
            if isinstance(result, JSONResponse):
                try:
                    body_content = json.loads(
                        result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                    )
                    return body_content.get("response", "An unknown error occurred.")
                except Exception as e:
                    logger.error(f"Could not parse JSONResponse body: {e}")
                    return "An unreadable error response was received from the bot."
            
            # Handle dict responses
            if isinstance(result, dict):
                response = result.get("response", str(result))
            else:
                response = str(result)
            
            cleaned_response = GeneralBotWrapper.clean_repetitive_response(response)
            return cleaned_response
            
        except Exception as e:
            logger.error(f"General bot error: {traceback.format_exc()}")
            return "I encountered an issue. Please try rephrasing your question about GoodBooks."

    @staticmethod
    def enhance_question_with_context(question: str, context: str, format_request: str = None) -> str:
        """Enhance question with context."""
        enhanced = f"Regarding GoodBooks Technologies ERP system: {question}"
        
        if format_request == "table":
            return f"{enhanced} Please provide in table format."
        elif format_request == "detailed":
            return f"{enhanced} Please provide detailed comprehensive information."
        
        return enhanced

    @staticmethod
    def clean_repetitive_response(response: str) -> str:
        """Clean repetitive phrases."""
        patterns = [
            r"As we previously discussed,?\s*",
            r"I'd be happy to help you with that!\s*",
            r"Based on our previous conversations?,?\s*"
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

# ===========================
# ENHANCED ORCHESTRATION AGENT
# ===========================
class EnhancedOrchestrationAgent:
    def __init__(self):
        """Initialize orchestration agent with Gemma 2B."""
        self.llm = ChatOllama(
            model=config.OLLAMA_MODEL,
            base_url=config.ollama_base_url,
            temperature=config.OLLAMA_TEMPERATURE
        )
        
        self.intent_patterns = {
            "general": {
                "keywords": ["company", "goodbooks", "employee", "policy", "security", 
                           "customer", "module", "what is", "tell me about", "explain"]
            },
            "formula": {
                "keywords": ["formula", "calculate", "compute", "sum", "average", 
                           "math", "equation", "total"]
            },
            "report": {
                "keywords": ["report", "data analysis", "analytics", "dashboard", 
                           "statistics", "metrics"]
            },
            "menu": {
                "keywords": ["menu", "navigation", "interface", "screen", "navigate", "find"]
            },
            "project": {
                "keywords": ["project", "project file", "project report", "task", "milestone"]
            }
        }
        
        # Initialize bot wrappers using BaseBotWrapper for consistency
        self.bots = {
            "general": GeneralBotWrapper(),
            "formula": BaseBotWrapper(formula_bot, "formula", FORMULA_BOT_AVAILABLE, "chat") if FORMULA_BOT_AVAILABLE else None,
            "report": BaseBotWrapper(report_bot, "report", REPORT_BOT_AVAILABLE, "report_chat") if REPORT_BOT_AVAILABLE else None,
            "menu": BaseBotWrapper(menu_bot, "menu", MENU_BOT_AVAILABLE, "chat") if MENU_BOT_AVAILABLE else None,
            "project": BaseBotWrapper(project_bot, "project", PROJECT_BOT_AVAILABLE, "project_chat") if PROJECT_BOT_AVAILABLE else None
        }
        
        self.conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent orchestrator for GoodBooks Technologies ERP system. 
Answer ONLY based on provided data sources. Available services: GeneralBot, FormulaBot, ReportBot, MenuBot, ProjectBot.

STRICT RULES:
1. Only answer questions about GoodBooks Technologies ERP system
2. If information is not available in GoodBooks data, say: "I don't have information about that in GoodBooks data."
3. Never answer questions outside GoodBooks scope (weather, sports, general knowledge, etc.)
4. Respect conversation continuity and context
5. Maintain topic continuity across conversation

Context: {context}
Previous Topic: {previous_topic}
Question: {question}"""),
            ("human", "{question}")
        ])

    def enhanced_intent_detection(self, question: str, context: str, 
                                 previous_memories: List[Dict], thread_id: str = None) -> tuple:
        """Detect intent with conversation continuity."""
        question_lower = question.lower().strip()
        
        # Check for greetings first
        if GreetingHandler.is_greeting(question):
            return "greeting", None, ""
        
        follow_up_indicators = [
            "complete detail", "full information", "elaborate", "table format",
            "about that", "more about", "tell me more", "how to use that",
            "can you explain", "what about", "regarding that"
        ]
        
        is_follow_up = any(indicator in question_lower for indicator in follow_up_indicators)
        
        # Maintain continuity if follow-up
        if is_follow_up and thread_id and previous_memories:
            most_recent_bot = None
            most_recent_topic = ""
            
            for memory in previous_memories:
                if (memory.get('thread_id') == thread_id and 
                    memory.get('bot_type') not in ['conversational', 'unavailable', 'greeting']):
                    most_recent_bot = memory.get('bot_type', 'general')
                    most_recent_topic = memory.get('topic', '')
                    break
            
            if most_recent_bot:
                logger.info(f"Maintaining {most_recent_bot} bot for follow-up")
                format_request = self._detect_format_request(question_lower)
                return most_recent_bot, format_request, most_recent_topic
        
        # New question - intent scoring
        intent_scores = {}
        for intent, config_data in self.intent_patterns.items():
            score = sum(2 if kw in question_lower else 0 for kw in config_data.get("keywords", []))
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            format_request = self._detect_format_request(question_lower)
            return best_intent, format_request, ""
        
        return "general", None, ""
    
    def _detect_format_request(self, question_lower: str) -> Optional[str]:
        """Detect if user wants specific format."""
        if any(word in question_lower for word in ["table", "tabular", "in table format"]):
            return "table"
        elif any(word in question_lower for word in ["complete", "full", "detailed", "comprehensive"]):
            return "detailed"
        return None

    async def process_request(self, username: str, question: str, thread_id: str, 
                             is_existing: bool = False):
        """Process chat request with thread isolation and validation."""
        
        # Handle greetings
        if GreetingHandler.is_greeting(question):
            greeting_response = GreetingHandler.get_response()
            await history_manager.add_message_to_thread(thread_id, question, greeting_response, "greeting")
            await enhanced_memory.add_memory(username, question, greeting_response, "greeting", thread_id)
            return {"response": greeting_response, "bot_type": "greeting", "thread_id": thread_id}
        
        # Get memories with thread isolation for existing threads
        recent_memories = await enhanced_memory.retrieve_memories(
            username, question, k=config.MEMORY_RETRIEVAL_COUNT, 
            thread_id=thread_id, thread_isolation=is_existing
        )
        
        context = self.build_context(username, question, thread_id, recent_memories, is_existing)
        
        # Validate GoodBooks context
        if not GoodBooksValidator.is_goodbooks_related(question, context):
            out_of_scope_response = GoodBooksValidator.get_out_of_scope_response()
            await history_manager.add_message_to_thread(thread_id, question, out_of_scope_response, "out_of_scope")
            await enhanced_memory.add_memory(username, question, out_of_scope_response, "out_of_scope", thread_id)
            return {"response": out_of_scope_response, "bot_type": "out_of_scope", "thread_id": thread_id}
        
        # Detect intent
        intent, format_request, previous_topic = self.enhanced_intent_detection(
            question, context, recent_memories, thread_id
        )
        
        logger.info(f"Routing to {intent}Bot for {username}")
        
        # Get bot response
        bot = self.bots.get(intent)
        if not bot:
            bot = self.bots["general"]
            logger.warning(f"Bot {intent} not available, falling back to general")
        
        simplified_context = f"Recent topic: {previous_topic}" if previous_topic else ""
        answer = await bot.answer(question, simplified_context, format_request)
        
        # Store in memory and thread
        await history_manager.add_message_to_thread(thread_id, question, answer, intent)
        await enhanced_memory.add_memory(username, question, answer, intent, thread_id)
        
        return {"response": answer, "bot_type": intent, "thread_id": thread_id}

    def build_context(self, username: str, query: str, thread_id: str, 
                     memories: List[Dict], is_isolated: bool) -> str:
        """Build conversation context."""
        context_parts = [f"User: {username}"]
        
        if memories:
            context_parts.append("Relevant conversations:")
            for mem in memories[:3]:
                context_parts.append(f"  - {mem.get('topic', 'Unknown')}: {mem.get('user_message', '')[:50]}...")
        
        return "\n".join(context_parts)

# Initialize orchestrator
enhanced_orchestrator = EnhancedOrchestrationAgent()

# ===========================
# FASTAPI APP
# ===========================
class ThreadRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class ThreadRenameRequest(BaseModel):
    thread_id: str
    new_title: str

app = FastAPI(
    title="GoodBooks Cloud RAG Chatbot",
    description="AI-powered chatbot for GoodBooks Technologies ERP system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# API ENDPOINTS
# ===========================

@app.post("/gbaiapi/chat")
async def chat_endpoint(request: ThreadRequest, login_details: LoginDTO = Depends(get_login_details)):
    """Main chat endpoint - creates new thread or continues conversation."""
    username = login_details.UserName
    user_input = request.message.strip()
    
    if not user_input:
        return JSONResponse(status_code=400, content={"error": "Message cannot be empty"})

    try:
        thread_id = request.thread_id
        is_existing = bool(thread_id)
        
        if not thread_id:
            thread_id = await history_manager.create_new_thread(username, user_input)
        
        result = await enhanced_orchestrator.process_request(username, user_input, thread_id, is_existing)
        
        # Save vectorstore in background
        asyncio.create_task(enhanced_memory.save_vectorstore())
        
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred."})

@app.post("/gbaiapi/thread_chat")
async def thread_chat(request: ThreadRequest, login_details: LoginDTO = Depends(get_login_details)):
    """Continue conversation in existing thread with strict isolation."""
    username = login_details.UserName
    thread_id = request.thread_id
    user_input = request.message.strip()
    
    if thread_id:
        thread = await history_manager.get_thread(thread_id)
        if not thread or thread.get("username") != username:
            return JSONResponse(status_code=404, content={"error": "Thread not found"})
    else:
        thread_id = await history_manager.create_new_thread(username, user_input)
    
    try:
        result = await enhanced_orchestrator.process_request(username, user_input, thread_id, is_existing=True)
        asyncio.create_task(enhanced_memory.save_vectorstore())
        return result
    except Exception as e:
        logger.error(f"Thread chat error: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred.", "thread_id": thread_id})

@app.get("/gbaiapi/threads")
async def get_threads(login_details: LoginDTO = Depends(get_login_details), limit: int = None):
    """Get user's conversation threads."""
    username = login_details.UserName
    threads = await history_manager.get_user_threads(username, limit)
    return {"username": username, "threads": threads, "total": len(threads)}

@app.get("/gbaiapi/thread/{thread_id}")
async def get_thread_details(thread_id: str, login_details: LoginDTO = Depends(get_login_details)):
    """Get thread details."""
    username = login_details.UserName
    thread = await history_manager.get_thread(thread_id)
    
    if not thread or thread.get("username") != username:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})
    
    return thread

@app.delete("/gbaiapi/thread/{thread_id}")
async def delete_thread(thread_id: str, login_details: LoginDTO = Depends(get_login_details)):
    """Delete thread."""
    username = login_details.UserName
    success = await history_manager.delete_thread(thread_id, username)
    
    if success:
        return {"message": "Thread deleted successfully"}
    else:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})

@app.put("/gbaiapi/thread/{thread_id}/rename")
async def rename_thread(thread_id: str, request: ThreadRenameRequest, login_details: LoginDTO = Depends(get_login_details)):
    """Rename thread."""
    username = login_details.UserName
    success = await history_manager.rename_thread(thread_id, username, request.new_title)
    
    if success:
        return {"message": "Thread renamed successfully"}
    else:
        return JSONResponse(status_code=404, content={"error": "Thread not found"})

@app.get("/gbaiapi/memory_search")
async def search_memories(query: str, login_details: LoginDTO = Depends(get_login_details), limit: int = 10, thread_id: str = None):
    """Search user memories."""
    username = login_details.UserName
    memories = await enhanced_memory.retrieve_memories(username, query, k=limit, thread_id=thread_id)
    
    return {
        "username": username,
        "query": query,
        "thread_id": thread_id,
        "found_memories": len(memories),
        "memories": memories
    }

@app.get("/gbaiapi/user_statistics")
async def get_user_statistics(login_details: LoginDTO = Depends(get_login_details)):
    """Get user statistics."""
    username = login_details.UserName
    
    # Get threads
    threads = await history_manager.get_user_threads(username, limit=1000)
    active_threads = [t for t in threads if t.get('is_active', True)]
    
    # Get memories from Firestore
    memories_query = enhanced_memory.memories_collection.where("username", "==", username).limit(1000)
    memories_stream = memories_query.stream()
    memories_list = [doc.to_dict() async for doc in memories_stream]
    
    # Calculate bot usage
    bot_usage = {}
    for memory in memories_list:
        bot_type = memory.get('bot_type', 'unknown')
        bot_usage[bot_type] = bot_usage.get(bot_type, 0) + 1
    
    # Recent activity
    now = datetime.now()
    recent_activity = {"today": 0, "this_week": 0, "this_month": 0}
    
    for memory in memories_list:
        try:
            mem_time = datetime.fromisoformat(memory.get('timestamp', ''))
            time_diff = now - mem_time
            
            if time_diff.days == 0:
                recent_activity["today"] += 1
            if time_diff.days <= 7:
                recent_activity["this_week"] += 1
            if time_diff.days <= 30:
                recent_activity["this_month"] += 1
        except:
            continue
    
    return {
        "username": username,
        "statistics": {
            "total_threads": len(threads),
            "active_threads": len(active_threads),
            "total_memories": len(memories_list),
            "bot_usage": bot_usage,
            "recent_activity": recent_activity
        },
        "generated_at": datetime.now().isoformat()
    }

@app.post("/gbaiapi/cleanup_old_data")
async def cleanup_old_data(login_details: LoginDTO = Depends(get_login_details), days_to_keep: int = None):
    """Cleanup old data (admin only)."""
    username = login_details.UserName
    
    if username != "admin":
        return JSONResponse(status_code=403, content={"error": "Unauthorized"})
    
    if days_to_keep is None:
        days_to_keep = config.DATA_RETENTION_DAYS
    
    try:
        # Cleanup old threads
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        # Delete old inactive threads
        old_threads_query = (history_manager.threads_ref
                           .where("is_active", "==", False)
                           .where("updated_at", "<", cutoff_date))
        old_threads_stream = old_threads_query.stream()
        
        deleted_count = 0
        async for doc in old_threads_stream:
            await doc.reference.delete()
            deleted_count += 1
        
        # Cleanup old memories
        old_memories_query = enhanced_memory.memories_collection.where("timestamp", "<", cutoff_date)
        old_memories_stream = old_memories_query.stream()
        
        memory_deleted_count = 0
        async for doc in old_memories_stream:
            await doc.reference.delete()
            memory_deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} threads and {memory_deleted_count} memories")
        
        return {
            "message": f"Cleaned up data older than {days_to_keep} days",
            "threads_deleted": deleted_count,
            "memories_deleted": memory_deleted_count,
            "cleanup_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return JSONResponse(status_code=500, content={"error": "Cleanup failed"})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vectorstore_loaded": enhanced_memory.is_loaded,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_url": config.ollama_base_url,
        "gcs_bucket": config.GCS_BUCKET_NAME
    }

@app.get("/config")
async def get_config(login_details: LoginDTO = Depends(get_login_details)):
    """Get current configuration (admin only)."""
    username = login_details.UserName
    
    if username != "admin":
        return JSONResponse(status_code=403, content={"error": "Unauthorized"})
    
    return {
        "gcs_bucket": config.GCS_BUCKET_NAME,
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_host": config.OLLAMA_HOST,
        "ollama_port": config.OLLAMA_PORT,
        "embedding_model": config.EMBEDDING_MODEL,
        "memory_save_frequency": config.MEMORY_SAVE_FREQUENCY,
        "memory_retrieval_count": config.MEMORY_RETRIEVAL_COUNT,
        "max_threads_per_user": config.MAX_THREADS_PER_USER,
        "data_retention_days": config.DATA_RETENTION_DAYS
    }

@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    logger.info("=" * 60)
    logger.info("Starting GoodBooks Cloud RAG Chatbot")
    logger.info(f"Model: {config.OLLAMA_MODEL}")
    logger.info(f"Ollama URL: {config.ollama_base_url}")
    logger.info(f"GCS Bucket: {config.GCS_BUCKET_NAME}")
    logger.info("=" * 60)
    await enhanced_memory.load_or_create_vectorstore()
    logger.info("Vector store loaded and ready.")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks."""
    logger.info("Shutting down...")
    await enhanced_memory.save_vectorstore()
    logger.info("Vector store saved. Goodbye!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=False)