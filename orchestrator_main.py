import json
import os
import logging
import traceback
import re
import uuid
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# If you use Firestore/GCS, these imports are optional
try:
    from google.cloud import firestore
    from google.cloud import storage
except Exception:
    firestore = None
    storage = None

import httpx

# ---------------------------
# Basic logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

# ---------------------------
# Optional GCP initialization
# ---------------------------
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
db = None
bucket = None
try:
    if GCP_PROJECT_ID and firestore:
        db = firestore.Client(project=GCP_PROJECT_ID)
    if GCS_BUCKET_NAME and storage:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"GCP config: project={GCP_PROJECT_ID} bucket={GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"GCP init failed (continuing without): {e}")

# ---------------------------
# Roles & greetings
# ---------------------------
class UserRole:
    DEVELOPER = "developer"
    IMPLEMENTATION = "implementation"
    MARKETING = "marketing"
    CLIENT = "client"
    ADMIN = "admin"

GREETING_PATTERNS = [
    r'^(hi|hello|hey|greetings|good morning|good afternoon|good evening|sup|yo|howdy)$',
    r'^(hi|hello|hey)\s+(there|everyone|all)$',
    r'^how are you\??$',
    r'^what\'?s up\??$'
]

ROLE_GREETINGS = {
    UserRole.DEVELOPER: "Hi! I'm your GoodBooks ERP technical assistant.",
    UserRole.IMPLEMENTATION: "Hello! I'm your GoodBooks implementation consultant.",
    UserRole.MARKETING: "Hi! I'm your GoodBooks product expert.",
    UserRole.CLIENT: "Hello! Welcome to GoodBooks ERP! 😊",
    UserRole.ADMIN: "Hello! I'm your GoodBooks system administrator assistant."
}

def is_greeting(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) > 4:
        return False
    for p in GREETING_PATTERNS:
        if re.search(p, t):
            return True
    return False

def get_greeting_response(user_role: str) -> str:
    return ROLE_GREETINGS.get(user_role, ROLE_GREETINGS[UserRole.CLIENT])

# ---------------------------
# Conversation threads
# ---------------------------
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
        if not db:
            logger.info("Firestore not configured - skipping thread load.")
            return
        try:
            logger.info("Loading threads from Firestore...")
            threads_ref = db.collection('conversation_threads')
            for doc in threads_ref.stream():
                data = doc.to_dict()
                if not data:
                    continue
                t = ConversationThread(data.get("thread_id"), data.get("username"), data.get("title"))
                t.created_at = data.get("created_at")
                t.updated_at = data.get("updated_at")
                t.messages = data.get("messages", [])
                t.is_active = data.get("is_active", True)
                self.threads[t.thread_id] = t
            logger.info(f"Loaded {len(self.threads)} threads from Firestore.")
        except Exception as e:
            logger.error(f"Failed to load threads from Firestore: {e}", exc_info=True)

    def save_threads(self):
        if not db:
            logger.info("Firestore not configured - skipping thread save.")
            return
        try:
            for thread_id, thread in self.threads.items():
                db.collection('conversation_threads').document(thread_id).set(thread.to_dict())
        except Exception as e:
            logger.error(f"Error saving threads to Firestore: {e}")

    def create_new_thread(self, username: str, initial_message: str = None) -> str:
        thread_id = str(uuid.uuid4())
        thread = ConversationThread(thread_id, username)
        if initial_message:
            thread.title = thread._generate_title_from_message(initial_message)
        self.threads[thread_id] = thread
        try:
            asyncio.create_task(asyncio.to_thread(self.save_threads))
        except Exception:
            pass
        logger.info(f"Created new thread {thread_id} for {username}")
        return thread_id

    def add_message_to_thread(self, thread_id: str, user_message: str, bot_response: str, bot_type: str):
        if thread_id in self.threads:
            self.threads[thread_id].add_message(user_message, bot_response, bot_type)
            try:
                asyncio.create_task(asyncio.to_thread(self.save_threads))
            except Exception:
                pass
    
    def get_user_threads(self, username: str, limit: int = 50) -> List[Dict]:
        user_threads = [
            t.to_dict() for t in self.threads.values()
            if t.username == username and t.is_active
        ]
        user_threads.sort(key=lambda x: x["updated_at"], reverse=True)
        return user_threads[:limit]
    
    def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        return self.threads.get(thread_id)
    
    def delete_thread(self, thread_id: str, username: str) -> bool:
        if thread_id in self.threads and self.threads[thread_id].username == username:
            self.threads[thread_id].is_active = False
            try:
                asyncio.create_task(asyncio.to_thread(self.save_threads))
            except Exception:
                pass
            return True
        return False
    
    def rename_thread(self, thread_id: str, username: str, new_title: str) -> bool:
        if thread_id in self.threads and self.threads[thread_id].username == username:
            self.threads[thread_id].title = new_title
            self.threads[thread_id].updated_at = datetime.now().isoformat()
            try:
                asyncio.create_task(asyncio.to_thread(self.save_threads))
            except Exception:
                pass
            return True
        return False
    
    def cleanup_old_threads(self, days_to_keep: int = 90):
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        threads_to_delete = []
        for thread_id, thread in list(self.threads.items()):
            if not thread.is_active and thread.updated_at < cutoff_iso:
                threads_to_delete.append(thread_id)
        if threads_to_delete:
            for thread_id in threads_to_delete:
                del self.threads[thread_id]
                if db:
                    db.collection('conversation_threads').document(thread_id).delete()
            logger.info(f"Cleaned up {len(threads_to_delete)} old threads")

history_manager = ConversationHistoryManager()

# ---------------------------
# Memory & embeddings
# ---------------------------
MEMORY_VECTORSTORE_PATH = "conversational_memory_vectorstore"
conversational_memory_metadata = {}
user_sessions = {}
chats_db = {}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 1}
    )
except Exception as e:
    logger.warning("HuggingFaceEmbeddings unavailable - memory disabled: %s", e)
    embeddings = None

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
            if bucket:
                faiss_blob = bucket.blob(f"{self.vectorstore_path}.faiss")
                pkl_blob = bucket.blob(f"{self.vectorstore_path}.pkl")
                if faiss_blob.exists() and pkl_blob.exists():
                    os.makedirs(self.vectorstore_path, exist_ok=True)
                    faiss_blob.download_to_filename(f"{self.vectorstore_path}.faiss")
                    pkl_blob.download_to_filename(f"{self.vectorstore_path}.pkl")
                    self.memory_vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                    logger.info("Loaded memory vectorstore from GCS.")
                    return
            if os.path.exists(f"{self.vectorstore_path}.faiss") and os.path.exists(f"{self.vectorstore_path}.pkl"):
                self.memory_vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded memory vectorstore from local disk.")
                return
        except Exception as e:
            logger.warning("Could not load memory vectorstore: %s", e)
        try:
            dummy_doc = Document(page_content="Memory system initialized")
            self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings) if self.embeddings else None
            logger.info("Created new memory vectorstore.")
        except Exception as e:
            logger.error("Failed initializing memory: %s", e)
            self.memory_vectorstore = None
    
    def store_conversation_turn(self, username: str, user_message: str, bot_response: str, bot_type: str, user_role: str, thread_id: str = None):
        try:
            if not self.memory_vectorstore:
                return
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
            conversational_memory_metadata[memory_id] = memory_doc.metadata
            self.memory_counter += 1
            if self.memory_counter % 20 == 0:
                try:
                    self.memory_vectorstore.save_local(self.vectorstore_path)
                    if bucket:
                        bucket.blob(f"{self.vectorstore_path}.faiss").upload_from_filename(f"{self.vectorstore_path}.faiss")
                        bucket.blob(f"{self.vectorstore_path}.pkl").upload_from_filename(f"{self.vectorstore_path}.pkl")
                except Exception as e:
                    logger.warning("Failed saving memory: %s", e)
        except Exception as e:
            logger.error(f"Error storing conversation turn: {e}")
    
    def retrieve_contextual_memories(self, username: str, query: str, k: int = 2, thread_id: str = None, thread_isolation: bool = False) -> List[Dict]:
        try:
            if not self.memory_vectorstore:
                return []
            docs = self.memory_vectorstore.similarity_search(query, k=k * 2)
            user_memories = {}
            for doc in docs:
                md = doc.metadata or {}
                if md.get("username") == username and md.get("memory_id") != "init":
                    if thread_isolation and thread_id and md.get("thread_id") != thread_id:
                        continue
                    mid = md.get("memory_id")
                    if mid not in user_memories:
                        user_memories[mid] = {
                            "memory_id": mid,
                            "timestamp": md.get("timestamp"),
                            "user_message": md.get("user_message"),
                            "bot_response": md.get("bot_response"),
                            "bot_type": md.get("bot_type"),
                            "user_role": md.get("user_role"),
                            "thread_id": md.get("thread_id"),
                            "content": doc.page_content
                        }
            sorted_memories = sorted(user_memories.values(), key=lambda x: x.get("timestamp", ""), reverse=True)
            return sorted_memories[:k]
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

enhanced_memory = EnhancedConversationalMemory(MEMORY_VECTORSTORE_PATH, "metadata.json", embeddings)

# ---------------------------
# HTTP bot endpoints configuration
# ---------------------------
BOT_ENDPOINTS = {
    "general": os.environ.get("GB_GENERAL_URL", "http://localhost:8085/gbaiapi/chat"),
    "formula": os.environ.get("GB_FORMULA_URL", "http://localhost:8084/gbaiapi/chat"),
    "menu": os.environ.get("GB_MENU_URL", "http://localhost:8083/gbaiapi/Menu-chat"),
    "report": os.environ.get("GB_REPORT_URL", "http://localhost:8082/gbaiapi/Report-chat"),
    "project": os.environ.get("GB_PROJECT_URL", "http://localhost:8081/gbaiapi/Project File-chat"),
}

_httpx_client = None
def get_httpx_client():
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=30.0)
    return _httpx_client

async def call_bot_via_http(bot_name: str, question: str, user_role: str, username: str, timeout: float = 20.0) -> Optional[str]:
    url = BOT_ENDPOINTS.get(bot_name)
    if not url:
        logger.error("No endpoint configured for bot: %s", bot_name)
        return None
    client = get_httpx_client()
    headers = {"Login": json.dumps({"UserName": username, "Role": user_role})}
    payload = {"content": question}
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            try:
                j = resp.json()
                if isinstance(j, dict) and "response" in j:
                    return j["response"]
                return json.dumps(j)
            except Exception:
                return resp.text
        else:
            logger.warning("Bot %s returned status %s: %s", bot_name, resp.status_code, resp.text[:200])
            return None
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.RequestError) as e:
        logger.error("HTTP call to bot %s failed: %s", bot_name, str(e))
        return None
    except Exception as e:
        logger.error("Unexpected error calling bot %s: %s", bot_name, str(e))
        return None

# simple wrappers that call above
class GeneralBotWrapper:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "orchestrator") -> Optional[str]:
        return await call_bot_via_http("general", question, user_role, username)

class FormulaBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "orchestrator") -> Optional[str]:
        return await call_bot_via_http("formula", question, user_role, username)

class ReportBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "orchestrator") -> Optional[str]:
        return await call_bot_via_http("report", question, user_role, username)

class MenuBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "orchestrator") -> Optional[str]:
        return await call_bot_via_http("menu", question, user_role, username)

class ProjectBot:
    @staticmethod
    async def answer(question: str, context: str, user_role: str, username: str = "orchestrator") -> Optional[str]:
        return await call_bot_via_http("project", question, user_role, username)

# ---------------------------
# Orchestration agent
# ---------------------------
class AIOrchestrationAgent:
    def __init__(self):
        try:
            self.routing_llm = ChatOllama(
                model=os.environ.get("ORCH_MODEL", "gemma:2b"),
                base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                temperature=0
            )
            self.response_llm = ChatOllama(
                model=os.environ.get("ORCH_MODEL", "gemma:2b"),
                base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
                temperature=0.3
            )
        except Exception as e:
            logger.warning("ChatOllama init failed in orchestrator: %s", e)
            self.routing_llm = None
            self.response_llm = None
        self.intent_cache: Dict[str, str] = {}

    def _get_cached_intent(self, question: str) -> Optional[str]:
        q = question.lower().strip()
        if any(w in q for w in ['calculate', 'compute', 'formula', 'sum', 'average', '+', '-', '*', '/', '=']):
            return "formula"
        if any(w in q for w in ['report', 'analyze', 'chart', 'data', 'dashboard', 'visualize', 'statistics']):
            return "report"
        if any(w in q for w in ['menu', 'navigate', 'where is', 'find screen', 'interface', 'how to access']):
            return "menu"
        if any(w in q for w in ['project', 'project file', 'project report', 'project document']):
            return "project"
        if q in self.intent_cache:
            return self.intent_cache[q]
        return None

    async def detect_intent_with_ai(self, question: str, context: str) -> str:
        cached_intent = self._get_cached_intent(question)
        if cached_intent:
            return cached_intent
        if not self.routing_llm:
            return "general"
        prompt = f"Route this query to ONE bot: general, formula, report, menu, project.\nQuery: {question}\nBot (one word):"
        try:
            resp = await asyncio.wait_for(self.routing_llm.ainvoke(prompt), timeout=4.0)
            intent = getattr(resp, "content", str(resp)).strip().lower()
            valid = ["general", "formula", "report", "menu", "project"]
            if intent not in valid:
                intent = "general"
            self.intent_cache[question.lower().strip()] = intent
            return intent
        except Exception as e:
            logger.warning("Intent detection fallback: %s", e)
            return self._get_cached_intent(question) or "general"

    async def generate_out_of_scope_response(self, question: str, user_role: str) -> str:
        if not self.response_llm:
            return "I'm your GoodBooks ERP assistant. I can help with GoodBooks-related questions. I don't have information about that topic."
        prompt = f"You are a GoodBooks ERP assistant ({user_role}). Politely decline if outside scope.\nQuestion: {question}\nResponse:"
        try:
            resp = await asyncio.wait_for(self.response_llm.ainvoke(prompt), timeout=6.0)
            return getattr(resp, "content", str(resp)).strip()
        except Exception:
            return "I'm your GoodBooks ERP assistant. I can help with GoodBooks system-related questions but not that topic."

    async def apply_role_perspective(self, answer: str, user_role: str, question: str) -> str:
        if len(answer) < 120:
            return answer
        if not self.response_llm:
            return answer
        role_personality = {
            "developer": "Answer briefly but with technical details and code examples when relevant.",
            "implementation": "Answer as a step-by-step implementer.",
            "marketing": "Answer emphasizing business value and ROI.",
            "client": "Answer in simple, non-technical language.",
            "admin": "Answer with system administration focus."
        }.get(user_role, "")
        prompt = f"{role_personality}\nQuestion: {question}\nAnswer: {answer}\nRewrite briefly for role {user_role} (tone only):"
        try:
            resp = await asyncio.wait_for(self.response_llm.ainvoke(prompt), timeout=8.0)
            role_adapted = getattr(resp, "content", str(resp)).strip()
            if role_adapted and len(role_adapted) > 20:
                return role_adapted
            return answer
        except Exception:
            return answer

    async def process_request(self, username: str, user_role: str, question: str,
                              thread_id: str = None, is_existing_thread: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        try:
            asyncio.create_task(asyncio.to_thread(update_user_session, username))
        except Exception:
            pass

        if is_greeting(question):
            greeting_response = get_greeting_response(user_role)
            try:
                asyncio.create_task(asyncio.to_thread(
                    enhanced_memory.store_conversation_turn,
                    username, question, greeting_response, "greeting", user_role, thread_id
                ))
            except Exception:
                pass
            if thread_id:
                try:
                    asyncio.create_task(asyncio.to_thread(history_manager.add_message_to_thread,
                                                         thread_id, question, greeting_response, "greeting"))
                except Exception:
                    pass
            return {"response": greeting_response, "bot_type": "greeting", "thread_id": thread_id, "user_role": user_role}

        if is_existing_thread and thread_id:
            recent_memories = enhanced_memory.retrieve_contextual_memories(username, question, k=2, thread_id=thread_id, thread_isolation=True)
            context = build_conversational_context(username, question, thread_id, thread_isolation=True)
        else:
            recent_memories = enhanced_memory.retrieve_contextual_memories(username, question, k=2, thread_id=thread_id, thread_isolation=False)
            context = build_conversational_context(username, question, thread_id, thread_isolation=False)

        intent = await self.detect_intent_with_ai(question, context)
        logger.info("Routing to bot: %s (question: %.60s)", intent, question)

        bot_answer = await call_bot_via_http(intent, question, user_role, username, timeout=25.0)

        if not bot_answer or len(str(bot_answer).strip()) < 6:
            logger.info("Bot returned empty/short answer; generating fallback")
            answer = await self.generate_out_of_scope_response(question, user_role)
            bot_type = "out_of_scope"
        else:
            answer = await self.apply_role_perspective(str(bot_answer), user_role, question)
            bot_type = intent

        try:
            asyncio.create_task(asyncio.to_thread(update_enhanced_memory,
                                                 username, question, answer, bot_type, user_role, thread_id))
        except Exception:
            pass

        elapsed = time.time() - start_time
        logger.info("Response completed in %.2fs (bot: %s)", elapsed, bot_type)
        return {"response": answer, "bot_type": bot_type, "thread_id": thread_id, "user_role": user_role}

ai_orchestrator = AIOrchestrationAgent()

# ---------------------------
# Helper functions
# ---------------------------
def update_user_session(username: str):
    try:
        now = datetime.now().isoformat()
        if not db:
            user_sessions[username] = {"last_activity": now, "session_count": 1}
            return
        session_ref = db.collection('user_sessions').document(username)
        doc = session_ref.get()
        if not doc.exists:
            data = {"first_seen": now, "last_activity": now, "session_count": 1, "total_interactions": 1}
        else:
            data = doc.to_dict()
            data["last_activity"] = now
            data["total_interactions"] = data.get("total_interactions", 0) + 1
        session_ref.set(data)
        user_sessions[username] = data
    except Exception as e:
        logger.warning("Error saving user session: %s", e)

def build_conversational_context(username: str, current_query: str, thread_id: str = None, thread_isolation: bool = False) -> str:
    parts = []
    sess = user_sessions.get(username, {})
    if sess:
        parts.append(f"User: {username}")
    if thread_isolation and thread_id:
        thread = history_manager.get_thread(thread_id)
        if thread and thread.messages:
            parts.append(f"Thread: {thread.title}")
            recent = thread.messages[-2:]
            parts.append("Recent:")
            for m in recent:
                parts.append(f"Q: {m['user_message'][:80]}")
                parts.append(f"A: {m['bot_response'][:80]}")
    else:
        if thread_id:
            thread = history_manager.get_thread(thread_id)
            if thread and thread.messages:
                recent = thread.messages[-1:]
                for m in recent:
                    parts.append(f"Q: {m['user_message'][:80]}")
                    parts.append(f"A: {m['bot_response'][:80]}")
    return "\n".join(parts)

def update_enhanced_memory(username: str, question: str, answer: str, bot_type: str, user_role: str, thread_id: str = None):
    try:
        if thread_id:
            history_manager.add_message_to_thread(thread_id, question, answer, bot_type)
        enhanced_memory.store_conversation_turn(username, question, answer, bot_type, user_role, thread_id)
    except Exception as e:
        logger.warning("Error storing memory: %s", e)

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="GoodBooks AI Orchestrator - FIXED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        proc = time.time() - start
        response.headers["X-Process-Time"] = f"{proc:.2f}s"
        if proc > 1.0:
            logger.info("Request %s took %.2fs", request.url.path, proc)
        return response

app.add_middleware(PerformanceMonitoringMiddleware)

class Message(BaseModel):
    content: str

class ThreadRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str

class ThreadRenameRequest(BaseModel):
    thread_id: str
    new_title: str

@app.post("/gbaiapi/chat", tags=["AI Role-Based Chat"])
async def ai_role_based_chat(message: Message, Login: str = Header(...)):
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client").lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header. Must include UserName and Role"})
    valid_roles = ["developer", "implementation", "marketing", "client", "admin"]
    if user_role not in valid_roles:
        return JSONResponse(status_code=400, content={"response": f"Invalid role. Must be one of: {', '.join(valid_roles)}"})
    user_input = message.content.strip()
    try:
        thread_id = await asyncio.to_thread(history_manager.create_new_thread, username, user_input)
        result = await ai_orchestrator.process_request(username, user_role, user_input, thread_id)
        logger.info("Response sent to %s (%s)", username, user_role)
        return result
    except Exception as e:
        logger.error("AI orchestration error: %s", e)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"response": "I encountered an error processing your request.", "bot_type": "error"})

@app.post("/gbaiapi/thread_chat", tags=["AI Thread Chat"])
async def ai_thread_chat(request: ThreadRequest, Login: str = Header(...)):
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client").lower()
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    valid_roles = ["developer", "implementation", "marketing", "client", "admin"]
    if user_role not in valid_roles:
        return JSONResponse(status_code=400, content={"response": f"Invalid role"})
    thread_id = request.thread_id
    user_input = request.message.strip()
    if thread_id:
        thread = history_manager.get_thread(thread_id)
        if not thread or thread.username != username:
            return JSONResponse(status_code=404, content={"response": "Thread not found"})
    else:
        thread_id = await asyncio.to_thread(history_manager.create_new_thread, username, user_input)
    try:
        result = await ai_orchestrator.process_request(username, user_role, user_input, thread_id, is_existing_thread=True)
        logger.info("Thread response sent to %s (%s)", username, user_role)
        return result
    except Exception as e:
        logger.error("Thread chat error: %s", e)
        return JSONResponse(status_code=500, content={"response": "I encountered an error. Please try again.", "bot_type": "error", "thread_id": thread_id})

@app.get("/gbaiapi/conversation_threads", tags=["Conversation History"])
async def get_conversation_threads(Login: str = Header(...), limit: int = 50):
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
        user_role = login_dto.get("Role", "client")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    threads = history_manager.get_user_threads(username, limit)
    session_info = user_sessions.get(username, {})
    return {"username": username, "user_role": user_role, "session_info": session_info, "threads": threads, "total_threads": len(threads)}

@app.get("/gbaiapi/thread/{thread_id}", tags=["Conversation History"])
async def get_thread_details(thread_id: str, Login: str = Header(...)):
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

@app.get("/gbaiapi/system_status", tags=["System Health"])
async def system_status():
    bot_status = {k: "configured" if v else "missing" for k, v in BOT_ENDPOINTS.items()}
    memory_stats = {
        "total_users": len(chats_db),
        "total_sessions": len(user_sessions),
        "total_conversations": sum(len(ch) for ch in chats_db.values()) if chats_db else 0,
        "total_memories": len(conversational_memory_metadata),
        "total_threads": len(history_manager.threads),
        "active_threads": len([t for t in history_manager.threads.values() if t.is_active])
    }
    return {"status": "healthy", "available_bots": list(BOT_ENDPOINTS.keys()), "bot_status": bot_status, "memory_system": memory_stats, "version": "7.0.0-FIXED"}

@app.post("/gbaiapi/cleanup_old_data", tags=["System Maintenance"])
async def cleanup_old_data(Login: str = Header(...), days_to_keep: int = 90):
    try:
        login_dto = json.loads(Login)
        user_role = login_dto.get("Role", "client")
        if user_role != "admin":
            return JSONResponse(status_code=403, content={"response": "Admin access required"})
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    try:
        await asyncio.to_thread(history_manager.cleanup_old_threads, days_to_keep)
        return {"message": f"Cleaned up data older than {days_to_keep} days", "cleanup_date": datetime.now().isoformat()}
    except Exception as e:
        logger.error("Cleanup error: %s", e)
        return JSONResponse(status_code=500, content={"response": "Cleanup failed"})

@app.get("/gbaiapi/debug/endpoint_test", tags=["Debug"])
async def endpoint_test():
    results = {}
    client = get_httpx_client()
    for name, url in BOT_ENDPOINTS.items():
        try:
            health_url = url.replace("/gbaiapi", "/gbaiapi/health")
            r = await client.get(health_url, timeout=5.0)
            results[name] = {"url": url, "status": r.status_code, "ok": r.status_code == 200, "text": (r.text[:300] if r.text else "")}
        except Exception as e:
            results[name] = {"url": url, "error": str(e)}
    return results

@app.on_event("startup")
async def startup_event():
    logger.info("Starting GoodBooks Orchestrator (fixed)...")
    if ai_orchestrator.routing_llm:
        try:
            asyncio.create_task(ai_orchestrator.routing_llm.ainvoke("ping"))
        except Exception as e:
            logger.warning("Routing LLM warmup scheduled but failed: %s", e)
    get_httpx_client()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down orchestrator...")
    try:
        if _httpx_client:
            await _httpx_client.aclose()
    except Exception:
        pass
    try:
        history_manager.save_threads()
    except Exception as e:
        logger.warning("Failed saving threads at shutdown: %s", e)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8010))
    logger.info(f"Starting orchestrator on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
