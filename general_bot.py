import os
import json
import logging
import traceback
import re
from typing import List, Dict
from datetime import datetime
# from langchain_ollama import ChatOllama
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Header
from fastapi import APIRouter
import pickle
 
# Load environment variables
load_dotenv()
 
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Paths
DOCUMENTS_DIR = "/app/data"
MEMORY_VECTORSTORE_PATH = "memory_vectorstore"
MEMORY_METADATA_FILE = "memory_metadata.json"
 
# Load chat history
 
# Load memory metadata
memory_metadata = {}
if os.path.exists(MEMORY_METADATA_FILE):
    with open(MEMORY_METADATA_FILE, "r") as f:
        memory_metadata = json.load(f)
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)
 
 
# Initialize LLM
# Get the Ollama URL from an environment variable, defaulting to localhost for local development

# llm = ChatOllama(
#     model="gemma:2b",
#     base_url="http://ollama:11434",
#     temperature=0.3
# )
 
# Initialize embeddings (shared for both document and memory vectorstores)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
class ConversationalMemory:
    def __init__(self, vectorstore_path: str, metadata_file: str, embeddings):
        self.vectorstore_path = vectorstore_path
        self.metadata_file = metadata_file
        self.embeddings = embeddings
        self.memory_vectorstore = None
        self.memory_counter = 0
       
        # Load existing memory vectorstore or create new one
        self.load_memory_vectorstore()
   
    def load_memory_vectorstore(self):
        """Load existing memory vectorstore or create a new one"""
        try:
            if os.path.exists(f"{self.vectorstore_path}.faiss"):
                logger.info("Loading existing memory vectorstore...")
                self.memory_vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # Get the current counter from metadata
                global memory_metadata
                self.memory_counter = len(memory_metadata)
                logger.info(f"Loaded memory vectorstore with {self.memory_counter} memories")
            else:
                logger.info("Creating new memory vectorstore...")
                # Create initial empty vectorstore with a dummy document
                dummy_doc = Document(
                    page_content="System initialized",
                    metadata={
                        "memory_id": "init",
                        "username": "system",
                        "timestamp": datetime.now().isoformat(),
                        "type": "system"
                    }
                )
                self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
                self.memory_vectorstore.save_local(self.vectorstore_path)
                logger.info("Created new memory vectorstore")
        except Exception as e:
            logger.error(f"Error loading memory vectorstore: {e}")
            # Fallback: create new vectorstore
            dummy_doc = Document(
                page_content="System initialized",
                metadata={
                    "memory_id": "init",
                    "username": "system",
                    "timestamp": datetime.now().isoformat(),
                    "type": "system"
                }
            )
            self.memory_vectorstore = FAISS.from_documents([dummy_doc], self.embeddings)
            self.memory_vectorstore.save_local(self.vectorstore_path)
   
    def add_conversation_turn(self, username: str, user_message: str, bot_response: str):
        """Add a conversation turn to memory vectorstore"""
        try:
            timestamp = datetime.now().isoformat()
            memory_id = f"{username}_{self.memory_counter}"
           
            # Create conversation context for better retrieval
            conversation_context = f"User: {user_message}\nAssistant: {bot_response}"
           
            # Create document for the conversation turn
            memory_doc = Document(
                page_content=conversation_context,
                metadata={
                    "memory_id": memory_id,
                    "username": username,
                    "timestamp": timestamp,
                    "user_message": user_message,
                    "bot_response": bot_response,
                    "type": "conversation"
                }
            )
           
            # Add to vectorstore
            self.memory_vectorstore.add_documents([memory_doc])
           
            # Update metadata
            global memory_metadata
            memory_metadata[memory_id] = {
                "username": username,
                "timestamp": timestamp,
                "user_message": user_message,
                "bot_response": bot_response
            }
           
            # Save vectorstore and metadata
            self.memory_vectorstore.save_local(self.vectorstore_path)
            self.save_metadata()
           
            self.memory_counter += 1
            logger.info(f"Added memory {memory_id} for user {username}")
           
        except Exception as e:
            logger.error(f"Error adding conversation turn to memory: {e}")
   
    def retrieve_relevant_memories(self, username: str, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant past conversations for the user"""
        try:
            if not self.memory_vectorstore:
                return []
           
            # Search for relevant memories
            docs = self.memory_vectorstore.similarity_search(
                query,
                k=k*2,  # Get more results to filter by user
                filter=None  # FAISS doesn't support metadata filtering directly
            )
           
            # Filter results for the specific user and exclude system messages
            user_memories = []
            for doc in docs:
                if (doc.metadata.get("username") == username and
                    doc.metadata.get("type") == "conversation"):
                    user_memories.append({
                        "timestamp": doc.metadata.get("timestamp"),
                        "user_message": doc.metadata.get("user_message"),
                        "bot_response": doc.metadata.get("bot_response"),
                        "content": doc.page_content
                    })
               
                if len(user_memories) >= k:
                    break
           
            logger.info(f"Retrieved {len(user_memories)} relevant memories for {username}")
            return user_memories
           
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
   
    def save_metadata(self):
        """Save memory metadata to file"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(memory_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memory metadata: {e}")
 
# Initialize conversational memory
conversational_memory = ConversationalMemory(
    MEMORY_VECTORSTORE_PATH,
    MEMORY_METADATA_FILE,
    embeddings
)
 
# Load and split documents
def load_text_and_json_files(documents_dir):
    # --- ADD THIS CHECK ---
    if not os.path.exists(documents_dir):
        logging.warning(f"Documents directory '{documents_dir}' not found. RAG will not be available.")
        return [] # Return an empty list if the directory doesn't exist
    # --- END OF ADDED CHECK ---

    all_docs = []
    files = [f for f in os.listdir(documents_dir) if f.endswith(('.txt', '.json'))]
    # ... rest of the function stays the same ...
    return all_docs
 
all_docs = load_text_and_json_files(DOCUMENTS_DIR)
text_chunks = None
retriever = None
 
if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Loaded {len(all_docs)} docs, split into {len(text_chunks)} chunks.")
 
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    logger.info("FAISS vectorstore and retriever initialized.")
else:
    logger.warning("No documents loaded. RAG will not be available.")
 
# Enhanced prompt template with memory integration
prompt_template = """
You are an expert assistant for GoodBooks Technologies. Answer the user's question using the provided context, recent conversation history, and relevant memories from past conversations.
 
Your role is to provide accurate, context-aware, and natural answers that build upon previous interactions and maintain conversational continuity.
 
Guidelines:
1. Use the provided company context ({context}) as the primary source of truth for factual information.
2. Consider the recent conversation history ({recent_chat_history}) for immediate context.
3. Use relevant memories from past conversations ({relevant_memories}) to maintain long-term conversational context and continuity.
   - Reference previous discussions when relevant
   - Build upon past conversations naturally
   - Remember user preferences and previous topics discussed
4. If the context does not contain the answer, rely on your general knowledge as a language model.
5. If you genuinely do not know the answer, say "I don't know" and suggest that the user ask something else.
6. Do not include reference numbers, citations, or system IDs in your response.
7. Keep your responses clear, concise, and helpful.
8. Maintain a natural conversational flow that acknowledges past interactions when relevant.
 
---
 
Relevant Memories from Past Conversations:
{relevant_memories}
 
Recent Conversation History:
{recent_chat_history}
 
Company Context:
{context}
 
Current User Question:
{question}
 
"""
 
class Message(BaseModel):
    content: str
 
 
async def call_hrms_tool(query: str):
    return "HRMS data not available due to connection error."
 
async def call_admin_tool(query: str):
    return "Checklist data not available due to connection error."
 
def spell_check(text: str):
    return text
 
def clean_response(text: str) -> str:
    cleaned_text = re.sub(r'\[[^\]]*\]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+([.,!?;:])', r'\1', cleaned_text)
    return cleaned_text.strip()
 
def format_as_points(text: str) -> str:
    points = re.split(r'\s*-\s+', text)
    points = [point.strip() for point in points if point.strip()]
    formatted_points = '\n'.join([f"- {point}" for point in points])
    return formatted_points
 
def format_memories(memories: List[Dict]) -> str:
    """Format retrieved memories for prompt"""
    if not memories:
        return "No relevant past conversations found."
   
    formatted = []
    for memory in memories:
        timestamp = memory.get("timestamp", "Unknown time")
        # Format timestamp to be more readable
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            readable_time = dt.strftime("%Y-%m-%d %H:%M")
        except:
            readable_time = timestamp
       
        formatted.append(f"[{readable_time}] {memory.get('content', '')}")
   
    return "\n".join(formatted)
 
@app.post("/gbaiapi/chat", tags=["Goodbooks Ai Api"])
async def chat(message: Message, Login: str = Header(...)):
    user_input = message.content.strip()
 
    # Parse login DTO from header
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
 
    user_input = spell_check(user_input)
 
    # Greeting check
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if user_input.lower() in greetings:
        # Even for greetings, check if we have past memories to provide personalized response
        relevant_memories = conversational_memory.retrieve_relevant_memories(username, user_input, k=2)
       
        if relevant_memories:
            formatted_answer = "Hello! Good to see you again. How can I help you today?"
        else:
            formatted_answer = "Hello! How can I help you today?"
       
       
        # Add to conversational memory
        conversational_memory.add_conversation_turn(username, user_input, formatted_answer)
       
        return {"response": formatted_answer}
 
    try:
        recent_chat_history_str = ""
 
        # Retrieve relevant memories from past conversations
        relevant_memories = conversational_memory.retrieve_relevant_memories(username, user_input, k=3)
        formatted_memories = format_memories(relevant_memories)
 
        # Get context from document retriever
        if retriever:
            docs = retriever.invoke(user_input)
            context_str = "\n".join([doc.page_content for doc in docs])
        else:
            context_str = ""
 
        # Create enhanced prompt with memory integration
        prompt_text = prompt_template.format(
            recent_chat_history=recent_chat_history_str,
            relevant_memories=formatted_memories,
            context=context_str,
            question=user_input
        )
       
        # Generate response
        answer = llm.invoke(prompt_text).content
 
        # Clean and format response
        cleaned_answer = clean_response(answer)
        formatted_answer = format_as_points(cleaned_answer)
 
 
        # Add conversation turn to long-term memory
        conversational_memory.add_conversation_turn(username, user_input, formatted_answer)
 
        return {"response": formatted_answer}
 
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"response": "An error occurred while processing your request."}
        )
 
 
@app.get("/gbaiapi/memory_stats", tags=["Goodbooks Ai Api"])
async def get_memory_stats(Login: str = Header(...)):
    """Get statistics about stored memories for the user"""
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
   
    # Count memories for this user
    user_memory_count = sum(1 for mem in memory_metadata.values() if mem.get("username") == username)
    total_memories = len(memory_metadata)
   
    return {
        "username": username,
        "user_memories": user_memory_count,
        "total_memories": total_memories,
        "memory_enabled": True
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8085)