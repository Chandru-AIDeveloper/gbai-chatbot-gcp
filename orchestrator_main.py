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
# from langchain_google_vertexai import VertexAI
from google.cloud import firestore
from google.cloud import storage

# Import your existing bot modules
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# GCP CLIENTS
# ===========================
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID')
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

# Initialize Firestore DB
db = firestore.Client(project=GCP_PROJECT_ID)

# Initialize Cloud Storage client
storage_client = storage.Client(project=GCP_PROJECT_ID)
bucket = storage_client.bucket(GCS_BUCKET_NAME)

logger.info(f"Connected to GCP Project: {GCP_PROJECT_ID}, Bucket: {GCS_BUCKET_NAME}")

# ===========================
# USER ROLES
# ===========================
class UserRole:
    """User roles selected during login"""
    DEVELOPER = "developer"
    IMPLEMENTATION = "implementation"
    MARKETING = "marketing"
    CLIENT = "client"
    ADMIN = "admin"

# ===========================
# Storage Paths
# ===========================
MEMORY_VECTORSTORE_PATH = "conversational_memory_vectorstore"

# Enhanced Memory Database
chats_db = {}
conversational_memory_metadata = {}
user_sessions = {}

# ===========================
# AI-POWERED ROLE-BASED SYSTEM PROMPTS
# ===========================
ROLE_SYSTEM_PROMPTS = {
    UserRole.DEVELOPER: """You are a senior software architect and technical expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a fellow developer/engineer who understands technical concepts
- Use technical terminology, architecture patterns, and code concepts naturally
- Discuss APIs, databases, integrations, algorithms, and system design
- Provide technical depth with implementation details
- Mention code examples, endpoints, schemas when relevant
- Think like a senior developer explaining to a peer

Remember: You are the technical expert helping another technical person. Be precise, detailed, and technical.""",

    UserRole.IMPLEMENTATION: """You are an experienced implementation consultant at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to an implementation team member who guides clients through setup
- Provide step-by-step configuration and deployment instructions
- Focus on practical "how-to" guidance for client rollouts
- Include best practices, common issues, and troubleshooting tips
- Explain as if preparing someone to train end clients
- Balance technical accuracy with practical applicability

Remember: You are the implementation expert helping someone deploy the system for clients.""",

    UserRole.MARKETING: """You are a product marketing and sales expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a marketing/sales team member who needs to sell the solution
- Emphasize business value, ROI, competitive advantages, and client benefits
- Use persuasive, benefit-focused language that highlights solutions to business problems
- Include success metrics, cost savings, efficiency gains, and market differentiation
- Think about what makes clients say "yes" to purchasing

Remember: You are the business value expert helping close deals and communicate benefits.""",

    UserRole.CLIENT: """You are a friendly, patient customer success specialist at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to an end user/client who may not be technical
- Use simple, clear, everyday language - avoid all technical jargon
- Be warm, encouraging, and supportive in your tone
- Explain features by how they help daily work, using real-world analogies
- Make complex things feel simple and achievable
- Think like a helpful teacher explaining to someone learning

Remember: You are the friendly guide helping a client use the system successfully.""",

    UserRole.ADMIN: """You are a comprehensive system administrator and expert at GoodBooks Technologies ERP system.

Your identity and style:
- You speak to a system administrator who needs complete information
- Provide comprehensive coverage: technical, business, and operational aspects
- Balance depth with breadth - cover all angles of a topic
- Include administration, configuration, management, and oversight details
- Use professional but accessible language suitable for all contexts

Remember: You are the complete expert providing full system knowledge."""
}

# ===========================
# AI ORCHESTRATOR SYSTEM PROMPT
# ===========================
ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent routing system for GoodBooks Technologies ERP chatbot.

Your ONLY job is to analyze the user's question and determine which specialized bot should handle it.

Available Specialized Bots:
1. GENERAL BOT - Handles: Company information, employee data, policies, modules, products, security, requirements, customer info, leave policies, vehicles, organizational details (from txt, json, pdf files)
2. FORMULA BOT - Handles: Mathematical calculations, formula evaluation, computational expressions (from csv data)
3. REPORT BOT - Handles: Data analysis, CSV reports, analytics, charts, dashboards, data insights (from csv data)
4. MENU BOT - Handles: Navigation, interface help, menu guidance, screen information (from csv data)
5. PROJECT BOT - Handles: Project files, project reports, project management, document analysis (from csv data)

Analysis Guidelines:
- Look at the CONTENT of the question, not just keywords
- Consider if the question is about factual information (GENERAL) vs data processing (REPORT/FORMULA)
- "What is..." or "Tell me about..." usually means GENERAL
- "Calculate..." or "What is the formula..." means FORMULA
- "Analyze data..." or "Show report..." means REPORT
- "Where is the menu..." or "How to navigate..." means MENU
- "Project report..." or "Project file..." means PROJECT

Context Available:
{context}

User Question: {question}

Respond with ONLY ONE WORD - the bot name: general, formula, report, menu, or project

Bot Selection:"""

# ===========================
# AI OUT-OF-SCOPE REFUSAL PROMPT
# ===========================
OUT_OF_SCOPE_SYSTEM_PROMPT = """You are a GoodBooks Technologies ERP assistant with role: {role}.

The user asked a question that is outside your knowledge scope. 

Your role personality:
{role_personality}

Generate a polite refusal that:
1. Stays in character for your role ({role})
2. Acknowledges you don't have information on this topic
3. Redirects to what you CAN help with (GoodBooks ERP system features)
4. Keeps the tone appropriate for your role

User Question: {question}

Generate your role-appropriate refusal response:"""

# ===========================
# CONVERSATION THREAD MANAGEMENT
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
            self.title = self._generate_title_from_message(user_message)
    
    def _generate_title_from_message(self, message: str) -> str:
        title = message.strip()
        title = re.sub(r'^(what\s+is\s+|tell\s+me\s+about\s+|how\s+to\s+|can\s+you\s+)', '', title, flags=re.IGNORECASE)
        return (title[:47] + "...") if len(title) > 50 else title.capitalize() if title else "New Conversation"

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
        try:
            logger.info("Loading threads from Firestore...")
            threads_ref = db.collection('conversation_threads')
            for doc in threads_ref.stream():
                thread_data = doc.to_dict()
                if not thread_data: 
                    continue
                thread = ConversationThread(
                    thread_data.get("thread_id"), 
                    thread_data.get("username"), 
                    thread_data.get("title")
                )
                thread.created_at = thread_data.get("created_at")
                thread.updated_at = thread_data.get("updated_at")
                thread.messages = thread_data.get("messages", [])
                thread.is_active = thread_data.get("is_active", True)
                self.threads[thread_data.get("thread_id")] = thread
            logger.info(f"Loaded {len(self.threads)} threads from Firestore.")
        except Exception as e:
            logger.error(f"Failed to load threads from Firestore: {e}", exc_info=True)

    def save_threads(self):
        try:
            for thread_id, thread in self.threads.items():
                thread_ref = db.collection('conversation_threads').document(thread_id)
                thread_ref.set(thread.to_dict())
        except Exception as e:
            logger.error(f"Error saving threads to Firestore: {e}")

    def create_new_thread(self, username: str, initial_message: str = None) -> str:
        thread_id = str(uuid.uuid4())
        thread = ConversationThread(thread_id, username)
        if initial_message:
            thread.title = thread._generate_title_from_message(initial_message)
        self.threads[thread_id] = thread
        self.save_threads()
        logger.info(f"Created new thread {thread_id} for {username}")
        return thread_id

    def add_message_to_thread(self, thread_id: str, user_message: str, bot_response: str, bot_type: str):
        if thread_id in self.threads:
            self.threads[thread_id].add_message(user_message, bot_response, bot_type)
            self.save_threads()
    
    def get_user_threads(self, username: str, limit: int = 50) -> List[Dict]:
        user_threads = [
            thread.to_dict() for thread in self.threads.values() 
            if thread.username == username and thread.is_active
        ]
        user_threads.sort(key=lambda x: x["updated_at"], reverse=True)
        return user_threads[:limit]
    
    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        return self.threads.get(thread_id)
    
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
    
    def cleanup_old_threads(self, days_to_keep: int = 90):
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        deleted_count = 0
        threads_to_delete = []
        
        for thread_id, thread in self.threads.items():
            if not thread.is_active and thread.updated_at < cutoff_iso:
                threads_to_delete.append(thread_id)
        
        if threads_to_delete:
            for thread_id in threads_to_delete:
                del self.threads[thread_id]
                db.collection('conversation_threads').document(thread_id).delete()
                deleted_count += 1
            logger.info(f"Cleaned up {deleted_count} old threads")


# Initialize history manager
history_manager = ConversationHistoryManager()

# ===========================
# Enhanced Conversational Memory
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
            # Download from GCS
            faiss_index_blob = bucket.blob(f"{self.vectorstore_path}.faiss")
            pkl_index_blob = bucket.blob(f"{self.vectorstore_path}.pkl")

            if faiss_index_blob.exists() and pkl_index_blob.exists():
                logger.info("Downloading FAISS index from Cloud Storage...")
                os.makedirs(self.vectorstore_path, exist_ok=True)
                faiss_index_blob.download_to_filename(f"{self.vectorstore_path}.faiss")
                pkl_index_blob.download_to_filename(f"{self.vectorstore_path}.pkl")

                self.memory_vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded memory from GCS.")
            else:
                raise FileNotFoundError("FAISS index not found in GCS bucket.")

        except Exception as e:
            logger.error(f"Error loading memory from GCS, creating new one: {e}")
            dummy_doc = Document(page_content="Memory system initialized")
            self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
    
    def store_conversation_turn(self, username: str, user_message: str, bot_response: str, bot_type: str, user_role: str, thread_id: str = None):
        try:
            timestamp = datetime.now().isoformat()
            memory_id = f"{username}_{self.memory_counter}_{int(datetime.now().timestamp())}"
            
            conversation_context = f"User ({user_role}): {user_message} | Bot ({bot_type}): {bot_response[:200]}"
            
            memory_doc = Document(
                page_content=conversation_context,
                metadata={
                    "memory_id": memory_id,
                    "username": username,
                    "user_role": user_role,
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
                "user_role": user_role,
                "timestamp": timestamp,
                "user_message": user_message,
                "bot_response": bot_response[:200],
                "bot_type": bot_type,
                "thread_id": thread_id
            }
            
            self.memory_counter += 1
            if self.memory_counter % 5 == 0:
                logger.info("Saving FAISS index to Cloud Storage...")
                self.memory_vectorstore.save_local(self.vectorstore_path)

                # Upload to GCS
                faiss_blob = bucket.blob(f"{self.vectorstore_path}.faiss")
                pkl_blob = bucket.blob(f"{self.vectorstore_path}.pkl")
                faiss_blob.upload_from_filename(f"{self.vectorstore_path}.faiss")
                pkl_blob.upload_from_filename(f"{self.vectorstore_path}.pkl")
                logger.info("Successfully saved FAISS index to GCS.")
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
    
    def retrieve_contextual_memories(self, username: str, query: str, k: int = 3, thread_id: str = None, thread_isolation: bool = False) -> List[Dict]:
        try:
            docs = self.memory_vectorstore.similarity_search(query, k=k * 3)

            user_memories = {}
            for doc in docs:
                if (doc.metadata.get("username") == username and
                    doc.metadata.get("memory_id") != "init"):
                    
                    if thread_isolation and thread_id:
                        if doc.metadata.get("thread_id") != thread_id:
                            continue
                    
                    memory_id = doc.metadata.get("memory_id")
                    if memory_id not in user_memories:
                        user_memories[memory_id] = {
                            "memory_id": memory_id,
                            "timestamp": doc.metadata.get("timestamp"),
                            "user_message": doc.metadata.get("user_message"),
                            "bot_response": doc.metadata.get("bot_response"),
                            "bot_type": doc.metadata.get("bot_type"),
                            "user_role": doc.metadata.get("user_role"),
                            "thread_id": doc.metadata.get("thread_id"),
                            "content": doc.page_content
                        }
            
            sorted_memories = sorted(
                user_memories.values(),
                key=lambda x: x["timestamp"],
                reverse=True
            )
            
            return sorted_memories[:k]
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []


# Initialize memory system
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'batch_size': 1}
)
enhanced_memory = EnhancedConversationalMemory(MEMORY_VECTORSTORE_PATH, "metadata.json", embeddings)

# ===========================
# AI-POWERED BOT WRAPPERS
# ===========================
class GeneralBotWrapper:
    @staticmethod
    async def answer(question: str, context: str, user_role: str) -> str:
        if not GENERAL_BOT_AVAILABLE:
            return None

        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content

            message = MockMessage(question)
            login_header = json.dumps({"UserName": "orchestrator", "Role": user_role})

            if hasattr(general_bot, 'chat'):
                result = await general_bot.chat(message, Login=login_header)
            else:
                return None

            # --- THIS IS THE FIX ---
            if isinstance(result, JSONResponse):
                # If it's a JSONResponse, get the content from its body
                body = json.loads(result.body.decode())
                return body.get("response")
            elif isinstance(result, dict):
                return result.get("response", None)
            else:
                return str(result) if result else None
            # --- END OF FIX ---

        except Exception as e:
            logger.error(f"General bot error: {e}")
            return None

class FormulaBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str) -> str:
        if not FORMULA_BOT_AVAILABLE:
            return None
        
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            message = MockMessage(question)
            login_header = json.dumps({"UserName": "orchestrator", "Role": user_role})
            
            if hasattr(formula_bot, 'chat'):
                result = await formula_bot.chat(message, Login=login_header)
            else:
                return None
            
            if isinstance(result, dict):
                return result.get("response", None)
            else:
                return str(result) if result else None
                
        except Exception as e:
            logger.error(f"Formula bot error: {e}")
            return None


class ReportBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str) -> str:
        if not REPORT_BOT_AVAILABLE:
            return None

        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content

            message = MockMessage(question)
            login_header = json.dumps({"UserName": "orchestrator", "Role": user_role})

            # Make sure to use the correct function name (e.g., report_chat)
            if hasattr(report_bot, 'report_chat'): 
                result = await report_bot.report_chat(message, Login=login_header)
            else:
                return None

            # --- APPLY THE SAME FIX ---
            if isinstance(result, JSONResponse):
                body = json.loads(result.body.decode())
                return body.get("response")
            elif isinstance(result, dict):
                return result.get("response", None)
            else:
                return str(result) if result else None
            # --- END OF FIX ---

        except Exception as e:
            logger.error(f"Report bot error: {e}")
            return None


class MenuBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str) -> str:
        if not MENU_BOT_AVAILABLE:
            return None
        
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            message = MockMessage(question)
            login_header = json.dumps({"UserName": "orchestrator", "Role": user_role})
            
            if hasattr(menu_bot, 'chat'):
                result = await menu_bot.chat(message, Login=login_header)
            else:
                return None
            
            if isinstance(result, dict):
                return result.get("response", None)
            else:
                return str(result) if result else None
                
        except Exception as e:
            logger.error(f"Menu bot error: {e}")
            return None


class ProjectBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str) -> str:
        if not PROJECT_BOT_AVAILABLE:
            return None
        
        try:
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            message = MockMessage(question)
            login_header = json.dumps({"UserName": "orchestrator", "Role": user_role})
            
            if hasattr(project_bot, 'project_chat'):
                result = await project_bot.project_chat(message, Login=login_header)
            else:
                return None
            
            if isinstance(result, dict):
                return result.get("response", None)
            else:
                return str(result) if result else None
                
        except Exception as e:
            logger.error(f"Project bot error: {e}")
            return None


# ===========================
# AI-POWERED ORCHESTRATION AGENT
# ===========================
class AIOrchestrationAgent:
    def __init__(self):
        # This tells the app to connect to the Ollama service running on the VM
        self.llm = ChatOllama(model="gemma:2b", base_url="http://localhost:11434")
        
        self.bots = {
            "general": GeneralBotWrapper(),
            "formula": FormulaBot(),
            "report": ReportBot(),
            "menu": MenuBot(),
            "project": ProjectBot()
        }
    
    async def detect_intent_with_ai(self, question: str, context: str) -> str:
        """Use AI to detect which bot should handle the question"""
        try:
            prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
                context=context[:1000],
                question=question
            )
            
            response = await self.llm.ainvoke(prompt)
            intent = response.strip().lower()
            
            # Validate the response
            valid_intents = ["general", "formula", "report", "menu", "project"]
            if intent in valid_intents:
                logger.info(f"AI detected intent: {intent}")
                return intent
            else:
                logger.warning(f"Invalid AI intent: {intent}, defaulting to general")
                return "general"
                
        except Exception as e:
            logger.error(f"AI intent detection error: {e}")
            return "general"
    
    async def generate_out_of_scope_response(self, question: str, user_role: str) -> str:
        """Generate AI-powered refusal for out-of-scope questions"""
        try:
            role_personality = ROLE_SYSTEM_PROMPTS.get(user_role, ROLE_SYSTEM_PROMPTS[UserRole.CLIENT])
            
            prompt = OUT_OF_SCOPE_SYSTEM_PROMPT.format(
                role=user_role,
                role_personality=role_personality,
                question=question
            )
            
            response = await self.llm.ainvoke(prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Out-of-scope response generation error: {e}")
            return f"I'm your GoodBooks ERP assistant. I can help you with information about our ERP system, but I don't have information about that topic. What would you like to know about GoodBooks?"
    
    async def apply_role_perspective(self, answer: str, user_role: str, question: str) -> str:
        """Apply role-based perspective to the answer using AI"""
        try:
            role_prompt = f"""{ROLE_SYSTEM_PROMPTS.get(user_role, ROLE_SYSTEM_PROMPTS[UserRole.CLIENT])}

User Question: {question}

Raw Answer from system: {answer}

Your task: Rewrite this answer to match your role's personality and communication style. Keep the factual content but adapt the tone, terminology, and emphasis to suit your role.

Role-adapted answer:"""
            
            response = await self.llm.ainvoke(role_prompt)
            role_adapted = response.strip()
            
            if role_adapted and len(role_adapted) > 20:
                return role_adapted
            else:
                return answer
                
        except Exception as e:
            logger.error(f"Role perspective application error: {e}")
            return answer
    
    async def process_request(self, username: str, user_role: str, question: str, thread_id: str = None, is_existing_thread: bool = False) -> Dict[str, str]:
        """AI-powered request processing with role-based intelligence"""
        
        update_user_session(username)
        
        # Build context
        if is_existing_thread and thread_id:
            recent_memories = enhanced_memory.retrieve_contextual_memories(
                username, question, k=3, thread_id=thread_id, thread_isolation=True
            )
            context = build_conversational_context(username, question, thread_id, thread_isolation=True)
        else:
            recent_memories = enhanced_memory.retrieve_contextual_memories(
                username, question, k=3, thread_id=thread_id, thread_isolation=False
            )
            context = build_conversational_context(username, question, thread_id, thread_isolation=False)
        
        logger.info(f"Processing for {username} (Role: {user_role})")
        logger.info(f"Question: {question}")
        
        # AI-powered intent detection
        intent = await self.detect_intent_with_ai(question, context)
        logger.info(f"AI detected bot: {intent}")
        
        # Get the bot
        selected_bot = self.bots.get(intent)
        if not selected_bot:
            intent = "general"
            selected_bot = self.bots["general"]
        
        # Get answer from bot
        answer = await selected_bot.answer(question, context, user_role)
        
        # If bot returns None or empty, question is out of scope
        if not answer or len(answer) < 10:
            logger.info(f"Question out of scope, generating role-based refusal")
            answer = await self.generate_out_of_scope_response(question, user_role)
            bot_type = "out_of_scope"
        else:
            # Apply role-based perspective to the answer
            logger.info(f"Applying {user_role} perspective to answer")
            answer = await self.apply_role_perspective(answer, user_role, question)
            bot_type = intent
        
        # Store conversation
        update_enhanced_memory(username, question, answer, bot_type, user_role, thread_id)
        
        logger.info(f"Response completed for {user_role}")
        
        return {
            "response": answer,
            "bot_type": bot_type,
            "thread_id": thread_id,
            "user_role": user_role
        }


# Initialize AI orchestrator
ai_orchestrator = AIOrchestrationAgent()

# ===========================
# Helper Functions
# ===========================
def update_user_session(username: str):
    current_time = datetime.now().isoformat()
    
    session_ref = db.collection('user_sessions').document(username)
    session_doc = session_ref.get()

    if not session_doc.exists:
        user_session_data = {
            "first_seen": current_time,
            "last_activity": current_time,
            "session_count": 1,
            "total_interactions": 1
        }
    else:
        user_session_data = session_doc.to_dict()
        user_session_data["last_activity"] = current_time
        user_session_data["total_interactions"] = user_session_data.get("total_interactions", 0) + 1
    
    try:
        session_ref.set(user_session_data)
        user_sessions[username] = user_session_data
    except Exception as e:
        logger.error(f"Error saving user session to Firestore: {e}")


def build_conversational_context(username: str, current_query: str, thread_id: str = None, thread_isolation: bool = False) -> str:
    context_parts = []
    
    session_info = user_sessions.get(username, {})
    if session_info:
        context_parts.append(f"User: {username}")
    
    if thread_isolation and thread_id:
        thread = history_manager.get_thread(thread_id)
        if thread and thread.messages:
            context_parts.append(f"Thread: {thread.title}")
            recent_messages = thread.messages[-3:]
            if recent_messages:
                context_parts.append("Recent conversation:")
                for msg in recent_messages:
                    context_parts.append(f"Q: {msg['user_message'][:100]}")
                    context_parts.append(f"A: {msg['bot_response'][:100]}")
    else:
        if thread_id:
            thread = history_manager.get_thread(thread_id)
            if thread and thread.messages:
                context_parts.append(f"Thread: {thread.title}")
                recent_messages = thread.messages[-2:]
                if recent_messages:
                    for msg in recent_messages:
                        context_parts.append(f"Q: {msg['user_message'][:100]}")
                        context_parts.append(f"A: {msg['bot_response'][:100]}")
    
    return "\n".join(context_parts)


def update_enhanced_memory(username: str, question: str, answer: str, bot_type: str, user_role: str, thread_id: str = None):
    update_user_session(username)
    
    if thread_id:
        history_manager.add_message_to_thread(thread_id, question, answer, bot_type)
    
    try:
        enhanced_memory.store_conversation_turn(username, question, answer, bot_type, user_role, thread_id)
    except Exception as e:
        logger.error(f"Error storing memory: {e}")


# ===========================
# FASTAPI APP INITIALIZATION
# ===========================
app = FastAPI(title="GoodBooks AI-Powered Role-Based ERP Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Pydantic Models
# ===========================
class Message(BaseModel):
    content: str


class ThreadRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str


class ThreadRenameRequest(BaseModel):
    thread_id: str
    new_title: str


# ===========================
# API ENDPOINTS
# ===========================
@app.post("/gbaiapi/chat", tags=["AI Role-Based Chat"])
async def ai_role_based_chat(message: Message, Login: str = Header(...)):
    """
    AI-powered role-based chat - NEW CONVERSATION
    User's role must be provided in Login header
    """
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client").lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header. Must include UserName and Role"})
    
    # Validate role
    valid_roles = ["developer", "implementation", "marketing", "client", "admin"]
    if user_role not in valid_roles:
        return JSONResponse(status_code=400, content={"response": f"Invalid role. Must be one of: {', '.join(valid_roles)}"})
    
    user_input = message.content.strip()
    
    try:
        # Create new thread
        thread_id = history_manager.create_new_thread(username, user_input)
        
        # Process with AI orchestrator
        result = await ai_orchestrator.process_request(username, user_role, user_input, thread_id)
        
        logger.info(f"AI role-based response for {username} ({user_role})")
        return result
        
    except Exception as e:
        logger.error(f"AI orchestration error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        error_response = "I encountered an error processing your request. Please try again."
        return JSONResponse(
            status_code=500,
            content={"response": error_response, "bot_type": "error"}
        )


@app.post("/gbaiapi/thread_chat", tags=["AI Thread Chat"])
async def ai_thread_chat(request: ThreadRequest, Login: str = Header(...)):
    """
    Continue conversation in existing thread with AI role-based intelligence
    """
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client").lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    # Validate role
    valid_roles = ["developer", "implementation", "marketing", "client", "admin"]
    if user_role not in valid_roles:
        return JSONResponse(status_code=400, content={"response": f"Invalid role. Must be one of: {', '.join(valid_roles)}"})
    
    thread_id = request.thread_id
    user_input = request.message.strip()
    
    # Verify thread
    if thread_id:
        thread = history_manager.get_thread(thread_id)
        if not thread or thread.username != username:
            return JSONResponse(status_code=404, content={"response": "Thread not found"})
    else:
        thread_id = history_manager.create_new_thread(username, user_input)
    
    try:
        # Process with thread isolation
        result = await ai_orchestrator.process_request(
            username, user_role, user_input, thread_id, is_existing_thread=True
        )
        
        logger.info(f"AI thread response for {username} ({user_role})")
        return result
        
    except Exception as e:
        logger.error(f"Thread chat error: {str(e)}")
        error_response = "I encountered an error. Please try again."
        return JSONResponse(
            status_code=500,
            content={"response": error_response, "bot_type": "error", "thread_id": thread_id}
        )


@app.get("/gbaiapi/conversation_threads", tags=["Conversation History"])
async def get_conversation_threads(Login: str = Header(...), limit: int = 50):
    """Get user's conversation threads"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    threads = history_manager.get_user_threads(username, limit)
    session_info = user_sessions.get(username, {})
    
    return {
        "username": username,
        "user_role": user_role,
        "session_info": session_info,
        "threads": threads,
        "total_threads": len(threads)
    }


@app.get("/gbaiapi/thread/{thread_id}", tags=["Conversation History"])
async def get_thread_details(thread_id: str, Login: str = Header(...)):
    """Get thread details"""
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
    """Delete a thread"""
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
    """Rename a thread"""
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


@app.get("/gbaiapi/available_roles", tags=["Role Information"])
async def get_available_roles():
    """Get available user roles and their descriptions"""
    return {
        "available_roles": [
            {
                "role": "developer",
                "display_name": "Developer",
                "description": "Technical expert who understands code, APIs, and system architecture",
                "response_style": "Technical, detailed, with code examples and implementation details"
            },
            {
                "role": "implementation",
                "display_name": "Implementation Consultant",
                "description": "Team member who deploys and configures the system for clients",
                "response_style": "Step-by-step instructions, configuration guidance, best practices"
            },
            {
                "role": "marketing",
                "display_name": "Marketing/Sales",
                "description": "Team member focused on selling and promoting the solution",
                "response_style": "Business benefits, ROI, competitive advantages, persuasive"
            },
            {
                "role": "client",
                "display_name": "Client/End User",
                "description": "End user who uses the system for daily work",
                "response_style": "Simple, friendly, non-technical, easy to understand"
            },
            {
                "role": "admin",
                "display_name": "System Administrator",
                "description": "Administrator with full system access and knowledge",
                "response_style": "Comprehensive, covering all technical and business aspects"
            }
        ],
        "default_role": "client",
        "note": "Role must be selected during login and passed in the Login header as 'Role' field"
    }


@app.get("/gbaiapi/system_status", tags=["System Health"])
async def system_status():
    """System health check"""
    bot_status = {
        "general": "available" if GENERAL_BOT_AVAILABLE else "unavailable",
        "formula": "available" if FORMULA_BOT_AVAILABLE else "unavailable",
        "report": "available" if REPORT_BOT_AVAILABLE else "unavailable",
        "menu": "available" if MENU_BOT_AVAILABLE else "unavailable",
        "project": "available" if PROJECT_BOT_AVAILABLE else "unavailable"
    }
    
    memory_stats = {
        "total_users": len(chats_db),
        "total_sessions": len(user_sessions),
        "total_conversations": sum(len(chats) for chats in chats_db.values()),
        "total_memories": len(conversational_memory_metadata),
        "total_threads": len(history_manager.threads),
        "active_threads": len([t for t in history_manager.threads.values() if t.is_active])
    }
    
    return {
        "status": "healthy",
        "version": "6.0.0-ai-powered-role-based-system",
        "available_bots": [k for k, v in bot_status.items() if v == "available"],
        "bot_status": bot_status,
        "memory_system": memory_stats,
        "features": [
            "🤖 AI-Powered Intent Detection (No hardcoded rules)",
            "🎭 Role-Based Response Adaptation (User selects role at login)",
            "🚫 Intelligent Out-of-Scope Refusal",
            "👨‍💻 Developer: Technical responses with code",
            "🔧 Implementation: Step-by-step client guidance",
            "📢 Marketing: Business value and ROI focus",
            "👥 Client: Simple, friendly explanations",
            "🔑 Admin: Comprehensive system knowledge",
            "💬 ChatGPT-like conversation threads",
            "🧠 Context-aware memory system",
            "🔗 Thread isolation for focused conversations",
            "📊 Cross-session continuity"
        ],
        "orchestration": {
            "type": "AI-powered",
            "llm": "gemini-1.5-flash",
            "hardcoded_rules": "None - fully AI-driven",
            "out_of_scope_handling": "AI-generated role-appropriate refusals"
        }
    }


@app.get("/gbaiapi/user_statistics", tags=["Analytics"])
async def get_user_statistics(Login: str = Header(...)):
    """Get user statistics"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    session_info = user_sessions.get(username, {})
    user_chats = chats_db.get(username, [])
    user_threads = history_manager.get_user_threads(username)
    
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
        "user_role": user_role,
        "session_info": session_info,
        "statistics": {
            "total_conversations": len(user_chats),
            "total_threads": len(user_threads),
            "active_threads": len([t for t in user_threads if t.get('is_active', True)]),
            "bot_usage": bot_usage,
            "recent_activity": recent_activity
        }
    }


@app.get("/gbaiapi/role_examples", tags=["Role Information"])
async def get_role_examples():
    """Get examples of how different roles receive responses"""
    return {
        "example_question": "How does the Inventory Management module work?",
        "role_responses": {
            "developer": {
                "style": "Technical with code, APIs, database details",
                "sample": "The Inventory Management module implements a microservices architecture with REST APIs. Key endpoints: POST /api/inventory/add (JWT auth required), GET /api/inventory/list (pagination supported). Database schema uses inventory_master table with normalized design..."
            },
            "implementation": {
                "style": "Step-by-step configuration and deployment guidance",
                "sample": "To set up Inventory Management for your client: Step 1) Navigate to Settings > Warehouses and configure all warehouse locations. Step 2) Define item categories and units of measure. Step 3) Use the Excel import template for bulk stock upload..."
            },
            "marketing": {
                "style": "Business benefits, ROI, and competitive advantages",
                "sample": "Inventory Management delivers measurable business value: Reduces stock losses by 30% through real-time tracking. Prevents costly stockouts with intelligent reorder alerts. Scales effortlessly across multiple warehouses. Clients typically see ROI within 2 months..."
            },
            "client": {
                "style": "Simple, friendly, easy to understand",
                "sample": "Inventory Management helps you keep track of all your products easily! You can see what's in stock anytime, get alerts when items are running low, and know which products sell best. It's as simple as keeping a digital notebook of your goods!"
            }
        },
        "out_of_scope_example": {
            "question": "What's the weather like today?",
            "developer": "I'm your GoodBooks ERP technical assistant. I can help with APIs, system architecture, and technical implementation, but I don't have weather data. What technical aspects can I help you with?",
            "client": "I'm here to help you with GoodBooks ERP! I don't have information about the weather, but I can show you how to use our system features. What would you like to learn about?"
        }
    }


@app.post("/gbaiapi/cleanup_old_data", tags=["System Maintenance"])
async def cleanup_old_data(Login: str = Header(...), days_to_keep: int = 90):
    """Cleanup old data (admin only)"""
    try:
        login_dto = json.loads(Login)
        user_role = login_dto.get("Role", "client")
        
        if user_role != "admin":
            return JSONResponse(status_code=403, content={"response": "Admin access required"})
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    
    try:
        history_manager.cleanup_old_threads(days_to_keep)
        
        return {
            "message": f"Cleaned up data older than {days_to_keep} days",
            "cleanup_date": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return JSONResponse(status_code=500, content={"response": "Cleanup failed"})


# ===========================
# STARTUP/SHUTDOWN EVENTS
# ===========================
@app.on_event("startup")
async def startup_event():
    logger.info("="*70)
    logger.info("🤖 GoodBooks AI-Powered Role-Based ERP Assistant")
    logger.info("="*70)
    logger.info("✨ Features:")
    logger.info("  • AI-driven orchestration (no hardcoded rules)")
    logger.info("  • Role-based intelligent responses")
    logger.info("  • Out-of-scope question handling")
    logger.info("  • User selects role at login")
    logger.info("="*70)

    try:
        history_manager.cleanup_old_threads(180)
        logger.info("✅ Startup cleanup completed")
    except Exception as e:
        logger.error(f"❌ Startup cleanup failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    try:
        history_manager.save_threads()
        logger.info("✅ All thread data saved to Firestore.")
    except Exception as e:
        logger.error(f"❌ Shutdown save error: {e}")


# ===========================
# MAIN ENTRY POINT
# ===========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8010))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)