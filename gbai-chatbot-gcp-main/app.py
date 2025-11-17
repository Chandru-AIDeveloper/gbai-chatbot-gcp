import json
import os
import logging
import traceback
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

# Import bot modules
try:
    import formula_bot
    FORMULA_BOT_AVAILABLE = True
    logging.info("Formula bot imported successfully")
except ImportError as e:
    FORMULA_BOT_AVAILABLE = False
    logging.warning(f"Formula bot not available: {e}")

try:
    import report_bot
    REPORT_BOT_AVAILABLE = True
    logging.info("Report bot imported successfully")
except ImportError as e:
    REPORT_BOT_AVAILABLE = False
    logging.warning(f"Report bot not available: {e}")

try:
    import menu_bot
    MENU_BOT_AVAILABLE = True
    logging.info("Menu bot imported successfully")
except ImportError as e:
    MENU_BOT_AVAILABLE = False
    logging.warning(f"Menu bot not available: {e}")

try:
    import project_bot
    PROJECT_BOT_AVAILABLE = True
    logging.info("Project bot imported successfully")
except ImportError as e:
    PROJECT_BOT_AVAILABLE = False
    logging.warning(f"Project bot not available: {e}")

try:
    import general_bot
    GENERAL_BOT_AVAILABLE = True
    logging.info("General bot imported successfully")
except ImportError as e:
    GENERAL_BOT_AVAILABLE = False
    logging.warning(f"General bot not available: {e}")

try:
    import schema_bot
    SCHEMA_BOT_AVAILABLE = True
    logging.info("Schema bot imported successfully")
except ImportError as e:
    SCHEMA_BOT_AVAILABLE = False
    logging.warning(f"Schema bot not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

MEMORY_VECTORSTORE_PATH = os.path.join(DATA_DIR, "conversational_memory_vectorstore")
MEMORY_METADATA_FILE = os.path.join(DATA_DIR, "conversational_memory_metadata.json")
USER_SESSIONS_FILE = os.path.join(DATA_DIR, "user_sessions.json")
CONVERSATION_THREADS_FILE = os.path.join(DATA_DIR, "conversation_threads.json")
ROLE_CONTEXT_FILE = os.path.join(DATA_DIR, "role_context_mapping.json")
CHATS_FILE = os.path.join(DATA_DIR, "chats.json") # Add this for chats.json

# ... later in the code, change "chats.json" to CHATS_FILE
# For example, in the "Memory Storage" section:
if os.path.exists(CHATS_FILE):
    with open(CHATS_FILE, "r", encoding='utf-8') as f:
        chats_db = json.load(f)

# ... and in the save_chats function:
def save_chats():
    try:
        with open(CHATS_FILE, "w", encoding='utf-8') as f:
            json.dump(chats_db, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chats: {e}")

# Performance optimizations
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

# ===========================
# Role-Based Access Configuration
# ===========================
role_context_mapping = {
    "ceo": {
        "access_level": "full",
        "focus_areas": ["strategic decisions", "company performance", "financial reports", "high-level analytics", "organizational structure", "all modules"],
        "context_priority": ["executive summary", "key metrics", "strategic insights"]
    },
    "sales": {
        "access_level": "sales_focused",
        "focus_areas": ["sales module", "crm", "customer data", "leads", "quotations", "orders", "pricing", "sales reports", "customer relationship"],
        "context_priority": ["sales data", "customer information", "quotation process", "order management"]
    },
    "marketing": {
        "access_level": "marketing_focused",
        "focus_areas": ["marketing campaigns", "customer analytics", "crm", "lead generation", "market trends", "brand management", "promotional activities"],
        "context_priority": ["marketing data", "customer insights", "campaign performance", "lead tracking"]
    },
    "hr": {
        "access_level": "hr_focused",
        "focus_areas": ["employee data", "payroll", "leave management", "recruitment", "performance management", "hr policies", "attendance", "employee benefits"],
        "context_priority": ["employee information", "leave policies", "payroll process", "hr procedures"]
    },
    "developer": {
        "access_level": "technical",
        "focus_areas": ["database schema", "technical documentation", "api integration", "system architecture", "development guidelines", "technical specifications"],
        "context_priority": ["technical details", "database structure", "api documentation", "system design"]
    },
    "finance": {
        "access_level": "finance_focused",
        "focus_areas": ["accounting", "financial reports", "budgets", "expenses", "revenue", "invoicing", "payments", "tax management", "financial analytics"],
        "context_priority": ["financial data", "accounting processes", "budget information", "payment details"]
    },
    "operations": {
        "access_level": "operations_focused",
        "focus_areas": ["inventory", "supply chain", "logistics", "warehouse management", "procurement", "production", "operations reports"],
        "context_priority": ["inventory data", "supply chain info", "logistics details", "operational metrics"]
    },
    "implementation": {
        "access_level": "implementation_focused",
        "focus_areas": ["project implementation", "system setup", "configuration", "training", "deployment", "client onboarding", "technical support"],
        "context_priority": ["implementation steps", "setup procedures", "configuration details", "training materials"]
    },
    "client": {
        "access_level": "client_limited",
        "focus_areas": ["product information", "how to use", "basic features", "support", "general queries", "user guides"],
        "context_priority": ["user guides", "basic features", "how-to information", "support details"]
    },
    "support": {
        "access_level": "support_focused",
        "focus_areas": ["customer support", "issue resolution", "troubleshooting", "user assistance", "technical support", "ticketing"],
        "context_priority": ["support procedures", "troubleshooting guides", "issue resolution", "customer assistance"]
    },
    "admin": {
        "access_level": "full",
        "focus_areas": ["all modules", "system administration", "user management", "permissions", "configurations", "security"],
        "context_priority": ["system settings", "user permissions", "security configurations", "all data"]
    }
}

# Load role context mapping
if os.path.exists(ROLE_CONTEXT_FILE):
    try:
        with open(ROLE_CONTEXT_FILE, "r") as f:
            loaded_mapping = json.load(f)
            role_context_mapping.update(loaded_mapping)
        logger.info(f"Loaded role context mapping for {len(role_context_mapping)} roles")
    except Exception as e:
        logger.warning(f"Could not load role context mapping: {e}, using defaults")
else:
    # Save default mapping
    try:
        with open(ROLE_CONTEXT_FILE, "w") as f:
            json.dump(role_context_mapping, f, indent=2)
        logger.info("Created default role context mapping file")
    except Exception as e:
        logger.warning(f"Could not save role context mapping: {e}")

# ===========================
# Memory Storage
# ===========================
chats_db = {}
conversational_memory_metadata = {}
user_sessions = {}
conversation_threads = {}

# Load existing data
try:
    if os.path.exists("chats.json"):
        with open("chats.json", "r", encoding='utf-8') as f:
            chats_db = json.load(f)

    if os.path.exists(MEMORY_METADATA_FILE):
        with open(MEMORY_METADATA_FILE, "r", encoding='utf-8') as f:
            conversational_memory_metadata = json.load(f)

    if os.path.exists(USER_SESSIONS_FILE):
        with open(USER_SESSIONS_FILE, "r", encoding='utf-8') as f:
            user_sessions = json.load(f)

    if os.path.exists(CONVERSATION_THREADS_FILE):
        with open(CONVERSATION_THREADS_FILE, "r", encoding='utf-8') as f:
            conversation_threads = json.load(f)
except json.JSONDecodeError as e:
    logger.error(f"Failed to load a JSON file: {e}. A new file will be created.")
except Exception as e:
    logger.error(f"An unexpected error occurred while loading data: {e}")

# ===========================
# Conversation Thread Management
# ===========================
class ConversationThread:
    def __init__(self, thread_id: str, username: str, title: str = None):
        self.thread_id = thread_id
        self.username = username
        self.title = title or "New Conversation"
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.messages = []
        self.is_active = True
        
    def add_message(self, user_message: str, bot_response: str, bot_type: str):
        message = {
            "id": str(uuid.uuid4()),
            "user_message": user_message,
            "bot_response": bot_response,
            "bot_type": bot_type,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        
        if self.title == "New Conversation" and len(self.messages) == 1:
            self.title = self.generate_title_from_message(user_message)
    
    def generate_title_from_message(self, message: str) -> str:
        title = message.strip()
        title = re.sub(r'^(what\s+is\s+|tell\s+me\s+about\s+|how\s+to\s+|can\s+you\s+)', '', title, flags=re.IGNORECASE)
        if len(title) > 50:
            title = title[:47] + "..."
        return title.capitalize() if title else "New Conversation"
    
    def get_full_conversation_history(self) -> str:
        if not self.messages:
            return ""
        
        history_parts = []
        for idx, msg in enumerate(self.messages, 1):
            history_parts.append(f"[Exchange {idx}]")
            history_parts.append(f"User: {msg['user_message']}")
            history_parts.append(f"Assistant ({msg['bot_type']}): {msg['bot_response']}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def to_dict(self) -> Dict:
        return {
            "thread_id": self.thread_id,
            "username": self.username,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
            "is_active": self.is_active,
            "message_count": len(self.messages)
        }

class ConversationHistoryManager:
    def __init__(self):
        self.threads = {}
        self.load_threads()
    
    def load_threads(self):
        global conversation_threads
        for thread_data in conversation_threads.values():
            thread = ConversationThread(
                thread_data["thread_id"],
                thread_data["username"],
                thread_data.get("title", "New Conversation")
            )
            thread.created_at = thread_data.get("created_at", thread.created_at)
            thread.updated_at = thread_data.get("updated_at", thread.updated_at)
            thread.messages = thread_data.get("messages", [])
            thread.is_active = thread_data.get("is_active", True)
            self.threads[thread.thread_id] = thread
        logger.info(f"Loaded {len(self.threads)} conversation threads")
    
    def create_new_thread(self, username: str, initial_message: str = None) -> str:
        thread_id = str(uuid.uuid4())
        title = "New Conversation"
        if initial_message:
            title = ConversationThread("", "").generate_title_from_message(initial_message)
        thread = ConversationThread(thread_id, username, title)
        self.threads[thread_id] = thread
        logger.info(f"Created new conversation thread {thread_id} for user {username}")
        return thread_id
    
    def add_message_to_thread(self, thread_id: str, user_message: str, bot_response: str, bot_type: str):
        if thread_id in self.threads:
            self.threads[thread_id].add_message(user_message, bot_response, bot_type)
            self.save_threads()
    
    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        return self.threads.get(thread_id)
    
    def get_user_threads(self, username: str, limit: int = 50) -> List[Dict]:
        user_threads = [
            thread.to_dict() for thread in self.threads.values()
            if thread.username == username and thread.is_active
        ]
        user_threads.sort(key=lambda x: x["updated_at"], reverse=True)
        return user_threads[:limit]
    
    def delete_thread(self, thread_id: str, username: str) -> bool:
        if thread_id in self.threads and self.threads[thread_id].username == username:
            self.threads[thread_id].is_active = False
            self.save_threads()
            return True
        return False
    
    def rename_thread(self, thread_id: str, username: str, new_title: str) -> bool:
        if thread_id in self.threads and self.threads[thread_id].username == username:
            self.threads[thread_id].title = new_title
            self.threads[thread_id].updated_at = datetime.now().isoformat()
            self.save_threads()
            return True
        return False
    
    def get_full_thread_context(self, thread_id: str) -> str:
        thread = self.threads.get(thread_id)
        return thread.get_full_conversation_history() if thread else ""
    
    def save_threads(self):
        global conversation_threads
        conversation_threads = {
            thread_id: thread.to_dict()
            for thread_id, thread in self.threads.items()
    }
        try:
            with open(CONVERSATION_THREADS_FILE, "w", encoding='utf-8') as f:
                json.dump(conversation_threads, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversation threads: {e}")
    
    def cleanup_old_threads(self, days_to_keep: int = 90):
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        deleted_count = 0
        for thread_id, thread in list(self.threads.items()):
            if not thread.is_active and thread.updated_at < cutoff_iso:
                del self.threads[thread_id]
                deleted_count += 1
        if deleted_count > 0:
            self.save_threads()
            logger.info(f"Cleaned up {deleted_count} old conversation threads")

history_manager = ConversationHistoryManager()

# ===========================
# Enhanced Memory System
# ===========================
class EnhancedConversationalMemory:
    def __init__(self, vectorstore_path: str, metadata_file: str, embeddings):
        self.vectorstore_path = vectorstore_path
        self.metadata_file = metadata_file
        self.embeddings = embeddings
        self.memory_vectorstore = None
        self.memory_counter = 0
        self.load_memory_vectorstore()
    
    def load_memory_vectorstore(self):
        try:
            if os.path.exists(f"{self.vectorstore_path}.faiss"):
                logger.info("Loading existing conversational memory vectorstore...")
                self.memory_vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.memory_counter = len(conversational_memory_metadata)
                logger.info(f"Loaded conversational memory with {self.memory_counter} memories")
            else:
                logger.info("Creating new conversational memory vectorstore...")
                dummy_doc = Document(
                    page_content="Conversational memory system initialized",
                    metadata={"memory_id": "system_init", "username": "system", "timestamp": datetime.now().isoformat()}
                )
                self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
                self.memory_vectorstore.save_local(self.vectorstore_path)
        except Exception as e:
            logger.error(f"Error loading conversational memory: {e}")
            dummy_doc = Document(
                page_content="Conversational memory system initialized",
                metadata={"memory_id": "system_init", "username": "system", "timestamp": datetime.now().isoformat()}
            )
            self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            self.memory_vectorstore.save_local(self.vectorstore_path)
    
    def store_conversation_turn(self, username: str, user_message: str, bot_response: str, bot_type: str, thread_id: str = None):
        try:
            timestamp = datetime.now().isoformat()
            memory_id = f"{username}_{self.memory_counter}_{int(datetime.now().timestamp())}"
            
            conversation_context = f"User: {user_message} | Bot ({bot_type}): {bot_response[:300]}..."
            
            memory_doc = Document(
                page_content=conversation_context,
                metadata={
                    "memory_id": memory_id,
                    "username": username,
                    "timestamp": timestamp,
                    "user_message": user_message,
                    "bot_response": bot_response[:500],
                    "bot_type": bot_type,
                    "thread_id": thread_id
                }
            )
            
            self.memory_vectorstore.add_documents([memory_doc])
            
            conversational_memory_metadata[memory_id] = {
                "username": username,
                "timestamp": timestamp,
                "user_message": user_message,
                "bot_response": bot_response[:300],
                "bot_type": bot_type,
                "thread_id": thread_id
            }
            
            self.memory_counter += 1
            if self.memory_counter % 5 == 0:
                self.memory_vectorstore.save_local(self.vectorstore_path)
                self.save_metadata()
            
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
    
    def retrieve_all_relevant_memories(self, username: str, current_query: str, k: int = 10, thread_id: str = None, thread_isolation: bool = False) -> List[Dict]:
        try:
            if not self.memory_vectorstore:
                return []
            
            docs = self.memory_vectorstore.similarity_search(current_query, k=k * 2, filter=None)
            
            user_memories = {}
            for doc in docs:
                if doc.metadata.get("username") == username and doc.metadata.get("memory_id") != "system_init":
                    if thread_isolation and thread_id and doc.metadata.get("thread_id") != thread_id:
                        continue
                    
                    memory_id = doc.metadata.get("memory_id")
                    if memory_id not in user_memories:
                        score = 1.0
                        if not thread_isolation and thread_id and doc.metadata.get("thread_id") == thread_id:
                            score += 0.5
                        
                        user_memories[memory_id] = {
                            "memory_id": memory_id,
                            "timestamp": doc.metadata.get("timestamp"),
                            "user_message": doc.metadata.get("user_message"),
                            "bot_response": doc.metadata.get("bot_response"),
                            "bot_type": doc.metadata.get("bot_type"),
                            "thread_id": doc.metadata.get("thread_id"),
                            "score": score
                        }
            
            sorted_memories = sorted(user_memories.values(), key=lambda x: (x["score"], x["timestamp"]), reverse=True)
            return sorted_memories[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving contextual memories: {e}")
            return []
    
    def save_metadata(self):
        try:
        # ... (code for cleaning metadata)
            with open(self.metadata_file, "w", encoding='utf-8') as f:
                json.dump(conversational_memory_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversational memory metadata: {e}")

    def cleanup_old_memories(self, days_to_keep: int = 90):
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_iso = cutoff_date.isoformat()
            old_entries = [memory_id for memory_id, metadata in conversational_memory_metadata.items() if metadata.get('timestamp', '') < cutoff_iso]
            for memory_id in old_entries:
                del conversational_memory_metadata[memory_id]
            if old_entries:
                logger.info(f"Cleaned up {len(old_entries)} old memory entries")
                self.save_metadata()
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")

# Initialize memory system
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'batch_size': 1}
)
enhanced_memory = EnhancedConversationalMemory(MEMORY_VECTORSTORE_PATH, MEMORY_METADATA_FILE, embeddings)

# ===========================
# Bot Wrapper Classes
# ===========================
class GeneralBotWrapper:
    @staticmethod
    async def answer(question: str, context: str) -> str:
        if not GENERAL_BOT_AVAILABLE:
            return "General information service is currently unavailable."
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            message = MockMessage(question)
            login_header = '{"UserName": "orchestrator"}'
            
            if hasattr(general_bot, 'chat'):
                result = await general_bot.chat(message, Login=login_header)
            else:
                return "General bot service temporarily unavailable."
            
            return GeneralBotWrapper._extract_response(result)
        except Exception as e:
            logger.error(f"General bot error: {e}\n{traceback.format_exc()}")
            return "I encountered an issue with the general information service."
    
    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, 'body'):
            try:
                body_content = result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                parsed = json.loads(body_content)
                return parsed.get("response", str(parsed))
            except:
                pass
        
        if isinstance(result, dict):
            return result.get("response", str(result))
        
        if isinstance(result, str):
            return result
        
        return str(result)

class FormulaBot:
    @staticmethod
    async def answer(question: str, context: str) -> str:
        if not FORMULA_BOT_AVAILABLE:
            return "Formula calculation service is currently unavailable."
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            message = MockMessage(question)
            login_header = '{"UserName": "orchestrator"}'
            
            if hasattr(formula_bot, 'chat'):
                result = await formula_bot.chat(message, Login=login_header)
            else:
                return "Formula bot service temporarily unavailable."
            
            return FormulaBot._extract_response(result)
        except Exception as e:
            logger.error(f"Formula bot error: {e}\n{traceback.format_exc()}")
            return "I encountered an issue with the formula calculation."
    
    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, 'body'):
            try:
                body_content = result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                parsed = json.loads(body_content)
                return parsed.get("response", str(parsed))
            except:
                pass
        
        if isinstance(result, dict):
            return result.get("response", str(result))
        
        if isinstance(result, str):
            return result
        
        return str(result)

class ReportBot:
    @staticmethod
    async def answer(question: str, context: str) -> str:
        if not REPORT_BOT_AVAILABLE:
            return "Report analysis service is currently unavailable."
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            message = MockMessage(question)
            login_header = '{"UserName": "orchestrator"}'
            
            if hasattr(report_bot, 'report_chat'):
                result = await report_bot.report_chat(message, Login=login_header)
            else:
                return "Report bot service temporarily unavailable."
            
            return ReportBot._extract_response(result)
        except Exception as e:
            logger.error(f"Report bot error: {e}\n{traceback.format_exc()}")
            return "I encountered an issue with the report analysis."
    
    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, 'body'):
            try:
                body_content = result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                parsed = json.loads(body_content)
                return parsed.get("response", str(parsed))
            except:
                pass
        
        if isinstance(result, dict):
            return result.get("response", str(result))
        
        if isinstance(result, str):
            return result
        
        return str(result)

class MenuBot:
    @staticmethod
    async def answer(question: str, context: str) -> str:
        if not MENU_BOT_AVAILABLE:
            return "Menu navigation service is currently unavailable."
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            message = MockMessage(question)
            login_header = '{"UserName": "orchestrator"}'
            
            if hasattr(menu_bot, 'chat'):
                result = await menu_bot.chat(message, Login=login_header)
            else:
                return "Menu bot service temporarily unavailable."
            
            return MenuBot._extract_response(result)
        except Exception as e:
            logger.error(f"Menu bot error: {e}\n{traceback.format_exc()}")
            return "I encountered an issue with the menu navigation."
    
    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, 'body'):
            try:
                body_content = result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                parsed = json.loads(body_content)
                return parsed.get("response", str(parsed))
            except:
                pass
        
        if isinstance(result, dict):
            return result.get("response", str(result))
        
        if isinstance(result, str):
            return result
        
        return str(result)

class ProjectBot:
    @staticmethod
    async def answer(question: str, context: str) -> str:
        if not PROJECT_BOT_AVAILABLE:
            return "Project management service is currently unavailable."
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            message = MockMessage(question)
            login_header = '{"UserName": "orchestrator"}'
            
            if hasattr(project_bot, 'project_chat'):
                result = await project_bot.project_chat(message, Login=login_header)
            else:
                return "Project bot service temporarily unavailable."
            
            return ProjectBot._extract_response(result)
        except Exception as e:
            logger.error(f"Project bot error: {e}\n{traceback.format_exc()}")
            return "I encountered an issue with the project analysis."
    
    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, 'body'):
            try:
                body_content = result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                parsed = json.loads(body_content)
                return parsed.get("response", str(parsed))
            except:
                pass
        
        if isinstance(result, dict):
            return result.get("response", str(result))
        
        if isinstance(result, str):
            return result
        
        return str(result)

class SchemaBot:
    @staticmethod
    async def answer(question: str, context: str) -> str:
        if not SCHEMA_BOT_AVAILABLE:
            return "Database schema service is currently unavailable."
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            message = MockMessage(question)
            login_header = '{"UserName": "orchestrator"}'
            
            if hasattr(schema_bot, 'chat'):
                result = await schema_bot.chat(message, Login=login_header)
            else:
                return "Schema bot service temporarily unavailable."
            
            return SchemaBot._extract_response(result)
        except Exception as e:
            logger.error(f"Schema bot error: {e}\n{traceback.format_exc()}")
            return "I encountered an issue with the database schema service."
    
    @staticmethod
    def _extract_response(result) -> str:
        if hasattr(result, 'body'):
            try:
                body_content = result.body.decode('utf-8') if isinstance(result.body, bytes) else result.body
                parsed = json.loads(body_content)
                return parsed.get("response", str(parsed))
            except:
                pass
        
        if isinstance(result, dict):
            return result.get("response", str(result))
        
        if isinstance(result, str):
            return result
        
        return str(result)

# ===========================
# Dynamic Smart Routing System
# ===========================
class DynamicSmartRouter:
    def __init__(self):
        self.embeddings = embeddings  # Reuse existing embeddings
        
        # Define what each bot knows about (their data domains)
        self.bot_capabilities = {
            "general": {
                "description": "Handles company information, employees, policies, products, modules, services, customers, vehicles, security, organizational details",
                "keywords": ["company", "employee", "policy", "product", "module", "service", "goodbooks", "customer", "vehicle", "security", "organization", "staff", "team", "department", "leave", "hr", "benefits", "crms", "erp", "software", "about", "what is", "tell me"]
            },
            "formula": {
                "description": "Performs mathematical calculations, formulas, expressions, computations",
                "keywords": ["calculate", "math", "formula", "compute", "addition", "subtraction", "multiplication", "division", "percentage", "sum", "total", "equation", "solve", "number", "digit", "+", "-", "*", "/", "="]
            },
            "report": {
                "description": "Analyzes data, generates reports, creates charts, provides analytics and statistics",
                "keywords": ["report", "analysis", "analytics", "chart", "graph", "dashboard", "statistics", "metrics", "data analysis", "performance", "insight", "visualization", "trend", "generate report"]
            },
            "menu": {
                "description": "Helps navigate application menus, screens, UI elements, interface locations",
                "keywords": ["menu", "navigate", "screen", "interface", "where is", "access", "location", "find", "ui", "page", "section", "how to access", "navigation"]
            },
            "project": {
                "description": "Manages project files, project analysis, project documents and resources",
                "keywords": ["project", "file", "document", "resource", "timeline", "task", "milestone", "planning", "project management"]
            },
            "schema": {
                "description": "Provides database schema information, table structures, columns, relationships, SQL DDL",
                "keywords": ["database", "table", "column", "schema", "sql", "field", "relation", "constraint", "index", "ddl", "structure", "tables", "columns"]
            }
        }
        
        self.llm = ChatOllama(model="gemma:2b", base_url="http://ollama:11434", temperature=0.2, num_ctx=6144)
        
        # Create routing prompt without hardcoded examples
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing assistant. Analyze the user's question and select the MOST APPROPRIATE bot.

Bot Capabilities:
{bot_info}

CRITICAL RULES:
1. Questions about company/products/modules/services → GENERAL bot
2. Questions about calculations/math → FORMULA bot  
3. Questions about data reports/analytics → REPORT bot
4. Questions about UI navigation → MENU bot
5. Questions about project files → PROJECT bot
6. Questions about database structure → SCHEMA bot

Format your response EXACTLY as:
SELECTED_BOT: [bot_name]
REASONING: [one sentence why]

Conversation History:
{conversation_history}

Current Question: {question}

Choose the bot:"""),
            ("human", "{question}")
        ])
    
    def get_bot_info_text(self) -> str:
        """Generate bot info without hardcoded examples"""
        info_parts = []
        for bot_name, capability in self.bot_capabilities.items():
            info_parts.append(f"- {bot_name.upper()}: {capability['description']}")
        return "\n".join(info_parts)
    
    async def route_intelligently(self, username: str, question: str, thread_id: str = None, is_existing_thread: bool = False) -> str:
        """Dynamically route based on question content"""
        
        question_lower = question.lower().strip()
        
        # Handle greetings
        if question_lower in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]:
            return "greeting"
        
        # STEP 1: Keyword-based initial scoring
        keyword_scores = {}
        for bot_name, capability in self.bot_capabilities.items():
            score = 0
            for keyword in capability["keywords"]:
                if keyword in question_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword.split())
            keyword_scores[bot_name] = score
        
        # STEP 2: Semantic similarity scoring
        semantic_scores = {}
        question_embedding = self.embeddings.embed_query(question)
        
        for bot_name, capability in self.bot_capabilities.items():
            desc_embedding = self.embeddings.embed_query(capability["description"])
            # Cosine similarity
            similarity = sum(a*b for a, b in zip(question_embedding, desc_embedding)) / (
                (sum(a*a for a in question_embedding) ** 0.5) * (sum(b*b for b in desc_embedding) ** 0.5)
            )
            semantic_scores[bot_name] = similarity
        
        # STEP 3: Combine scores
        combined_scores = {}
        for bot_name in self.bot_capabilities.keys():
            # Weighted combination: keywords (50%) + semantic (50%)
            combined_scores[bot_name] = (keyword_scores[bot_name] * 0.5) + (semantic_scores[bot_name] * 0.5)
        
        # Get top candidates
        sorted_bots = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_bot = sorted_bots[0][0]
        
        logger.info(f"🎯 Routing Scores for '{question}':")
        for bot, score in sorted_bots:
            logger.info(f"   {bot}: {score:.3f} (keywords: {keyword_scores[bot]}, semantic: {semantic_scores[bot]:.3f})")
        
        # STEP 4: If scores are close, use LLM as tiebreaker
        if len(sorted_bots) > 1 and (sorted_bots[0][1] - sorted_bots[1][1]) < 0.2:
            logger.info(f"⚖️ Close scores detected, using LLM for final decision...")
            try:
                context = self.build_context(username, question, thread_id, is_existing_thread)
                
                chain = self.routing_prompt | self.llm | StrOutputParser()
                llm_response = await chain.ainvoke({
                    "bot_info": self.get_bot_info_text(),
                    "conversation_history": context["conversation_history"],
                    "question": question
                })
                
                logger.info(f"🤖 LLM Decision:\n{llm_response}")
                
                llm_choice = self.parse_routing_response(llm_response)
                logger.info(f"✅ Final Routing: {llm_choice} (LLM tiebreaker)")
                return llm_choice
            except Exception as e:
                logger.error(f"LLM routing failed: {e}")
                logger.info(f"✅ Final Routing: {top_bot} (fallback to score-based)")
                return top_bot
        else:
            logger.info(f"✅ Final Routing: {top_bot} (clear winner by score)")
            return top_bot
    
    def parse_routing_response(self, response: str) -> str:
        """Parse LLM response to extract bot name"""
        response_lower = response.lower()
        
        if "selected_bot:" in response_lower:
            for line in response.split('\n'):
                if "selected_bot:" in line.lower():
                    bot_name = line.split(':', 1)[1].strip().lower()
                    bot_name = bot_name.replace('[', '').replace(']', '').strip()
                    if bot_name in self.bot_capabilities:
                        return bot_name
        
        # Fallback: search for bot names in response
        for bot_name in self.bot_capabilities.keys():
            if bot_name in response_lower:
                return bot_name
        
        logger.warning(f"Could not parse routing decision, defaulting to general")
        return "general"
    
    def build_context(self, username: str, question: str, thread_id: str = None, is_existing_thread: bool = False) -> Dict[str, str]:
        """Build conversation context"""
        conversation_history = ""
        
        if is_existing_thread and thread_id:
            conversation_history = history_manager.get_full_thread_context(thread_id)
        else:
            today = datetime.now().date()
            user_chats = chats_db.get(username, [])
            today_chats = []
            
            for chat in user_chats[-20:]:
                try:
                    chat_date = datetime.fromisoformat(chat['timestamp']).date()
                    if chat_date == today:
                        today_chats.append(chat)
                except:
                    continue
            
            if today_chats:
                history_parts = []
                for idx, chat in enumerate(today_chats[-5:], 1):
                    history_parts.append(f"User: {chat['user']}")
                    history_parts.append(f"Bot: {chat['bot'][:100]}...")
                conversation_history = "\n".join(history_parts)
        
        return {"conversation_history": conversation_history}


# ===========================
# Smart Orchestration Agent
# ===========================
class SmartOrchestrationAgent:
    def __init__(self):
        self.router = DynamicSmartRouter()
        
        self.bots = {
            "general": GeneralBotWrapper(),
            "formula": FormulaBot(),
            "report": ReportBot(),
            "menu": MenuBot(),
            "project": ProjectBot(),
            "schema": SchemaBot()
        }
    
    async def route_with_intelligence(self, username: str, question: str, thread_id: str = None, is_existing_thread: bool = False) -> Dict[str, str]:
        """Route using dynamic intelligent routing"""
        
        # Check for greeting
        question_lower = question.lower().strip()
        if question_lower in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]:
            answer = f"Hello! I'm your GoodBooks Technologies assistant. How can I help you today?"
            update_memory(username, question, answer, "greeting", thread_id)
            return {"response": answer, "bot_type": "greeting", "thread_id": thread_id}
        
        try:
            # Get dynamic routing decision
            intent = await self.router.route_intelligently(username, question, thread_id, is_existing_thread)
            
            # Build context for the selected bot
            context = self.router.build_context(username, question, thread_id, is_existing_thread)
            
            # Get answer from selected bot
            selected_bot = self.bots.get(intent, self.bots["general"])
            raw_answer = await selected_bot.answer(question, context["conversation_history"])
            
            # Clean the response to remove meta-commentary
            answer = clean_bot_response(raw_answer)
            
            # Store in memory
            update_memory(username, question, answer, intent, thread_id)
            
            return {"response": answer, "bot_type": intent, "thread_id": thread_id}
            
        except Exception as e:
            logger.error(f"Orchestration error: {e}\n{traceback.format_exc()}")
            answer = "I encountered an error. Please try again."
            update_memory(username, question, answer, "error", thread_id)
            return {"response": answer, "bot_type": "error", "thread_id": thread_id}


# Initialize the orchestrator
smart_orchestrator = SmartOrchestrationAgent()

# ===========================
# Helper Functions
# ===========================
def clean_bot_response(response: str) -> str:
    """Remove meta-commentary from bot responses to make them ChatGPT-like"""
    if not response:
        return response
    
    # Remove common prefixes
    prefixes_to_remove = [
        "sure, here is the answer to the user's question:",
        "here is the answer to the user's question:",
        "here's the answer to the user's question:",
        "here is the answer:",
        "here's the answer:",
        "the answer is:",
        "sure, here is the answer:",
        "sure, here's the answer:",
        "based on the context,",
        "according to the information,",
        "as per the data,",
    ]
    
    response_lower = response.lower().strip()
    
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            # Remove the prefix and clean up
            response = response[len(prefix):].strip()
            # Remove leading quotes if present
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1].strip()
            break
    
    # Remove bullet/dash at the start if it's a single response
    if response.startswith('- '):
        response = response[2:].strip()
    
    # Remove leading quotes
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()
    
    return response

def update_user_session(username: str):
    current_time = datetime.now().isoformat()
    if username not in user_sessions:
        user_sessions[username] = {
            "first_seen": current_time,
            "last_activity": current_time,
            "session_count": 1,
            "total_interactions": 0
        }
    else:
        user_sessions[username]["last_activity"] = current_time
        user_sessions[username]["total_interactions"] += 1
    
    try:
        with open(USER_SESSIONS_FILE, "w", encoding='utf-8') as f:
            json.dump(user_sessions, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving user sessions: {e}")

def save_chats():
    try:
        with open("chats.json", "w", encoding='utf-8') as f:
            json.dump(chats_db, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chats: {e}")

def update_memory(username: str, question: str, answer: str, bot_type: str, thread_id: str = None):
    update_user_session(username)
    
    if username not in chats_db:
        chats_db[username] = []
    
    chats_db[username].append({
        "user": question,
        "bot": answer,
        "bot_type": bot_type,
        "timestamp": datetime.now().isoformat(),
        "thread_id": thread_id
    })
    
    save_chats()
    
    if thread_id:
        history_manager.add_message_to_thread(thread_id, question, answer, bot_type)
    
    try:
        enhanced_memory.store_conversation_turn(username, question, answer, bot_type, thread_id)
    except Exception as e:
        logger.error(f"Error storing in long-term memory: {e}")

# ===========================
# FastAPI Setup
# ===========================
class Message(BaseModel):
    content: str

class ThreadRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str

class ThreadRenameRequest(BaseModel):
    thread_id: str
    new_title: str

app = FastAPI(title="Smart GoodBooks Orchestration Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# API Endpoints
# ===========================
@app.post("/gbaiapi/chat", tags=["Chat"])
async def chat(message: Message, Login: str = Header(...)):
    """New conversation with dynamic routing"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    user_input = message.content.strip()
    
    try:
        thread_id = history_manager.create_new_thread(username, user_input)
        result = await smart_orchestrator.route_with_intelligence(username, user_input, thread_id)
        result["thread_id"] = thread_id
        return result
    except Exception as e:
        logger.error(f"Chat error: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"response": "Error processing request", "bot_type": "error"})

@app.post("/gbaiapi/thread_chat", tags=["Thread Chat"])
async def thread_chat(request: ThreadRequest, Login: str = Header(...)):
    """Continue conversation in thread"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    thread_id = request.thread_id
    user_input = request.message.strip()
    
    if thread_id:
        thread = history_manager.get_thread(thread_id)
        if not thread or thread.username != username:
            return JSONResponse(status_code=404, content={"response": "Thread not found"})
    else:
        thread_id = history_manager.create_new_thread(username, user_input)
    
    try:
        result = await smart_orchestrator.route_with_intelligence(username, user_input, thread_id, is_existing_thread=True)
        result["thread_id"] = thread_id
        return result
    except Exception as e:
        logger.error(f"Thread chat error: {str(e)}")
        return JSONResponse(status_code=500, content={"response": "Error processing request", "bot_type": "error", "thread_id": thread_id})

@app.get("/gbaiapi/conversation_threads", tags=["Conversation History"])
async def get_conversation_threads(Login: str = Header(...), limit: int = 50):
    """Get all user's conversation threads"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    threads = history_manager.get_user_threads(username, limit)
    session_info = user_sessions.get(username, {})
    
    return {
        "username": username,
        "session_info": session_info,
        "threads": threads,
        "total_threads": len(threads)
    }

@app.get("/gbaiapi/thread/{thread_id}", tags=["Conversation History"])
async def get_thread_details(thread_id: str, Login: str = Header(...)):
    """Get complete thread details"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    thread = history_manager.get_thread(thread_id)
    
    if not thread or thread.username != username:
        return JSONResponse(status_code=404, content={"response": "Thread not found"})
    
    return thread.to_dict()

@app.delete("/gbaiapi/thread/{thread_id}", tags=["Conversation History"])
async def delete_thread(thread_id: str, Login: str = Header(...)):
    """Delete a conversation thread"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    success = history_manager.delete_thread(thread_id, username)
    
    if success:
        return {"message": "Thread deleted successfully"}
    else:
        return JSONResponse(status_code=404, content={"response": "Thread not found"})

@app.put("/gbaiapi/thread/{thread_id}/rename", tags=["Conversation History"])
async def rename_thread(thread_id: str, request: ThreadRenameRequest, Login: str = Header(...)):
    """Rename a conversation thread"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    success = history_manager.rename_thread(thread_id, username, request.new_title)
    
    if success:
        return {"message": "Thread renamed successfully"}
    else:
        return JSONResponse(status_code=404, content={"response": "Thread not found"})

@app.get("/gbaiapi/conversation_history", tags=["Legacy Support"])
async def get_conversation_history(Login: str = Header(...), limit: int = 20):
    """Get user's conversation history"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    user_chats = chats_db.get(username, [])[-limit:]
    session_info = user_sessions.get(username, {})
    user_memories = [mem for mem in conversational_memory_metadata.values() if mem.get("username") == username]
    threads = history_manager.get_user_threads(username, limit)
    
    return {
        "username": username,
        "session_info": session_info,
        "recent_conversations": user_chats,
        "total_conversations": len(chats_db.get(username, [])),
        "total_memories": len(user_memories),
        "threads": threads,
        "total_threads": len(threads)
    }

@app.get("/gbaiapi/memory_search", tags=["Memory Management"])
async def search_memories(query: str, Login: str = Header(...), limit: int = 10, thread_id: str = None):
    """Search user's memories"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    relevant_memories = enhanced_memory.retrieve_all_relevant_memories(username, query, k=limit, thread_id=thread_id)
    
    return {
        "username": username,
        "query": query,
        "thread_id": thread_id,
        "found_memories": len(relevant_memories),
        "memories": relevant_memories
    }

@app.post("/gbaiapi/cleanup_old_data", tags=["System Maintenance"])
async def cleanup_old_data(Login: str = Header(...), days_to_keep: int = 90):
    """Cleanup old conversation data"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        
        if username != "admin":
            return JSONResponse(status_code=403, content={"response": "Unauthorized"})
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    try:
        history_manager.cleanup_old_threads(days_to_keep)
        enhanced_memory.cleanup_old_memories(days_to_keep)
        
        return {
            "message": f"Successfully cleaned up data older than {days_to_keep} days",
            "cleanup_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return JSONResponse(status_code=500, content={"response": "Cleanup failed"})

@app.get("/gbaiapi/user_statistics", tags=["Analytics"])
async def get_user_statistics(Login: str = Header(...)):
    """Get user statistics"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    session_info = user_sessions.get(username, {})
    user_chats = chats_db.get(username, [])
    user_threads = history_manager.get_user_threads(username)
    user_memories = [mem for mem in conversational_memory_metadata.values() if mem.get("username") == username]
    
    bot_usage = {}
    for chat in user_chats:
        bot_type = chat.get('bot_type', 'unknown')
        bot_usage[bot_type] = bot_usage.get(bot_type, 0) + 1
    
    now = datetime.now()
    recent_activity = {"today": 0, "this_week": 0, "this_month": 0}
    
    for chat in user_chats:
        try:
            chat_time = datetime.fromisoformat(chat['timestamp'])
            time_diff = now - chat_time
            
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
        "session_info": session_info,
        "statistics": {
            "total_conversations": len(user_chats),
            "total_threads": len(user_threads),
            "active_threads": len([t for t in user_threads if t.get('is_active', True)]),
            "total_memories": len(user_memories),
            "bot_usage": bot_usage,
            "recent_activity": recent_activity
        },
        "generated_at": datetime.now().isoformat()
    }

@app.post("/gbaiapi/export_conversations", tags=["Data Export"])
async def export_conversations(Login: str = Header(...), format: str = "json"):
    """Export user's conversation data"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    if format not in ["json", "csv"]:
        return JSONResponse(status_code=400, content={"response": "Format must be 'json' or 'csv'"})
    
    try:
        user_threads = history_manager.get_user_threads(username)
        user_chats = chats_db.get(username, [])
        session_info = user_sessions.get(username, {})
        
        export_data = {
            "username": username,
            "export_date": datetime.now().isoformat(),
            "session_info": session_info,
            "threads": user_threads,
            "legacy_chats": user_chats
        }
        
        if format == "json":
            return export_data
        else:
            return {"message": "CSV export feature coming soon", "data": export_data}
    except Exception as e:
        logger.error(f"Export error: {e}")
        return JSONResponse(status_code=500, content={"response": "Export failed"})

@app.get("/gbaiapi/health", tags=["System"])
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "routing": "Dynamic keyword + semantic + LLM tiebreaker",
        "memory_system": "Full conversation history",
        "timestamp": datetime.now().isoformat(),
        "available_bots": {
            "general": GENERAL_BOT_AVAILABLE,
            "formula": FORMULA_BOT_AVAILABLE,
            "report": REPORT_BOT_AVAILABLE,
            "menu": MENU_BOT_AVAILABLE,
            "project": PROJECT_BOT_AVAILABLE,
            "schema": SCHEMA_BOT_AVAILABLE
        }
    }

@app.get("/gbaiapi/system_stats", tags=["System"])
async def system_stats(Login: str = Header(...)):
    """System statistics"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        
        if username != "admin":
            return JSONResponse(status_code=403, content={"response": "Unauthorized"})
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    return {
        "total_users": len(user_sessions),
        "total_threads": len(conversation_threads),
        "total_memories": len(conversational_memory_metadata),
        "total_chats": sum(len(chats) for chats in chats_db.values()),
        "active_threads": sum(1 for t in conversation_threads.values() if t.get('is_active', True)),
        "model_info": {
            "routing_method": "Keyword + Semantic + LLM tiebreaker",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": "gemma:2b",
            "memory_system": "Full conversation history"
        },
        "timestamp": datetime.now().isoformat()
    }

# ===========================
# Startup and Shutdown Events
# ===========================
@app.on_event("startup")
async def startup_event():
    logger.info("="*70)
    logger.info("🚀 Starting Smart GoodBooks Orchestration Chatbot")
    logger.info("="*70)
    logger.info("📊 Routing: Dynamic Keyword + Semantic + LLM Tiebreaker")
    logger.info("🧠 Memory: Full conversation history retrieval")
    logger.info("🤖 No hardcoded routing - adapts to ANY question")
    logger.info("="*70)
    try:
        history_manager.cleanup_old_threads(180)
        enhanced_memory.cleanup_old_memories(180)
        logger.info("✅ Initial data cleanup completed")
    except Exception as e:
        logger.error(f"❌ Initial cleanup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down chatbot...")
    try:
        save_chats()
        history_manager.save_threads()
        enhanced_memory.save_metadata()
        logger.info("✅ All data saved successfully")
    except Exception as e:
        logger.error(f"❌ Error saving data during shutdown: {e}")

# ===========================
# Run Server
# ===========================
if __name__ == "__main__":
    import uvicorn
    logger.info("="*70)
    logger.info("🚀 Smart GoodBooks Orchestration Chatbot Starting...")
    logger.info("="*70)
    logger.info("Features:")
    logger.info("  ✅ Dynamic routing (keyword + semantic analysis)")
    logger.info("  ✅ LLM tiebreaker for close decisions")
    logger.info("  ✅ Full conversation history memory")
    logger.info("  ✅ No hardcoded question examples")
    logger.info("  ✅ Adapts to ANY user question dynamically")
    logger.info("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8010)