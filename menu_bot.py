import json
import os
import logging
import traceback
import pandas as pd
from langchain_ollama import ChatOllama
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from fastapi import Header
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
DOCUMENTS_DIR = "data/"
 
class Message(BaseModel):
    content: str
 
def spell_check(text: str) -> str:
    # Placeholder for spell check - you can integrate a spell checker here
    return text
 
def clean_response(text: str) -> str:
    # Clean up response formatting
    text = text.strip()
    # Remove excessive newlines
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text
 
def format_as_points(text: str) -> str:
    # Keep natural paragraph format instead of forcing bullet points
    return text
 
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)
 
# Get the Ollama URL from an environment variable, defaulting to localhost for local development
llm = ChatOllama(
    model="gemma:2b",
    base_url="http://localhost:11434",
    temperature=0.3
)
 
def load_csv_as_document(file_path: str) -> List[Document]:
    """Load CSV file and convert to documents"""
    docs = []
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
       
        # Convert CSV to a more readable format
        csv_content = f"Data from {os.path.basename(file_path)}:\n\n"
       
        # Add column information
        csv_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
       
        # Convert each row to a readable format
        for idx, row in df.iterrows():
            row_content = f"Record {idx + 1}:\n"
            for col in df.columns:
                row_content += f"- {col}: {row[col]}\n"
            row_content += "\n"
           
            # Create a document for each row or group of rows
            doc = Document(
                page_content=row_content,
                metadata={"source": file_path, "row_index": idx}
            )
            docs.append(doc)
       
        # Also create a summary document with overall structure
        summary_content = f"Dataset Summary for {os.path.basename(file_path)}:\n"
        summary_content += f"Total records: {len(df)}\n"
        summary_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
       
        # Add sample data
        summary_content += "Sample data:\n"
        summary_content += df.head().to_string(index=False)
       
        summary_doc = Document(
            page_content=summary_content,
            metadata={"source": file_path, "type": "summary"}
        )
        docs.append(summary_doc)
       
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
       
    return docs
 
# Enhanced document loading function
def load_and_split_documents(documents_dir: str) -> List[Document]:
    all_docs = []
    
    menu_docs_path = os.path.join(documents_dir, "menu.csv")
    if os.path.exists(menu_docs_path):
        logger.info(f"Loading file: {menu_docs_path}")
        csv_docs = load_csv_as_document(menu_docs_path)
        all_docs.extend(csv_docs)
        logger.info(f"Loaded {len(csv_docs)} documents from CSV: {menu_docs_path}")
    else:
        logger.warning(f"{menu_docs_path} not found.")

    return all_docs
 
# Load documents
all_docs = load_and_split_documents(DOCUMENTS_DIR)
text_chunks = None
retriever = None
 
if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased chunk size for better context
        chunk_overlap=300  # Increased overlap
    )
    text_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Loaded {len(all_docs)} docs, split into {len(text_chunks)} chunks.")
   
    # Also fix the deprecated HuggingFaceEmbeddings import
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.warning("Using deprecated HuggingFaceEmbeddings. Consider upgrading: pip install langchain-huggingface")
   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Retrieve more context
    logger.info("FAISS vectorstore and retriever initialized.")
else:
    logger.warning("No documents loaded. RAG will not be available.")
 
# Updated prompt template for more natural responses
prompt_template = """
[ROLE]
You are an expert Menu assistant for GoodBooks Technologies.

[TASK]
Answer user questions related to Menu in a natural and conversational way.

[CONTEXT]
Use the Menu context provided below to find the answer:
{context}

[CONVERSATION HISTORY]
{history}

[REASONING]
- First, analyze the given Menu context carefully.
- If the answer exists in the context, summarize it clearly.
- If the answer is missing, do not invent information.

[OUTPUT]
Provide a clear, concise, and professional answer in natural language that maintains conversational continuity.

[CONDITIONS]
- If the context does not contain the answer, reply only with:
  "I don't know. Please try asking a different Menu-related question."
- Never hallucinate or add extra unrelated details.

[USER QUESTION]
{question}


Response:"""
 
prompt = ChatPromptTemplate.from_template(prompt_template)
 
if retriever:
    chain = (
        {
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
 
@app.post("/gbaiapi/Menu-chat", tags=["Goodbooks Ai Api"])
async def chat(message: Message, Login: str = Header(...)):
    user_input = message.content.strip()
   
    # Parse login DTO from header
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
   
    user_input = spell_check(user_input)
   
    # Enhanced greeting detection
    greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "what's up", "sup"
    ]
    if any(greeting in user_input.lower() for greeting in greetings):
        formatted_answer = "Hello! I'm here to help you with any questions you have. I can assist you with information from the available data sources. What would you like to know?"
       
        return {"response": formatted_answer}
   
    try:
        history_str = ""
       
        if retriever:
            answer = chain.invoke({"question": user_input, "history": history_str})
        else:
            # Fallback when no documents are loaded
            system_prompt = """You are a helpful AI assistant. Provide natural, conversational responses to user questions.
            Be friendly, informative, and honest about what you can and cannot help with."""
           
            full_prompt = f"{system_prompt}\n\nConversation history:\n{history_str}\nHuman: {user_input}\nAssistant:"
            answer = llm.invoke(full_prompt).content
       
        cleaned_answer = clean_response(answer)
        formatted_answer = format_as_points(cleaned_answer)
       
        # Store conversation
       
        return {"response": formatted_answer}
       
    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"response": "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."}
        )
 
 
@app.get("/gbaiapi/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "documents_loaded": len(all_docs) if all_docs else 0,
        "chunks_created": len(text_chunks) if text_chunks else 0,
        "retriever_available": retriever is not None
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)