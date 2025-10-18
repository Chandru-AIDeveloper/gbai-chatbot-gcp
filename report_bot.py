import json
import os
import logging
import traceback
import pandas as pd
# from langchain_ollama import ChatOllama
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
    return text  # Placeholder (can add real spell checker later)

def clean_response(text: str) -> str:
    text = text.strip()
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')
    return text

def format_as_points(text: str) -> str:
    return text


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the Ollama URL from an environment variable, defaulting to localhost for local development
# llm = ChatOllama(
#     model="gemma:2b",
#     base_url="http://ollama:11434",
#     temperature=0.3
# )

def load_csv_as_document(file_path: str) -> List[Document]:
    """Load CSV file and convert to documents"""
    docs = []
    try:
        df = pd.read_csv(file_path, encoding='cp1252')

        # Convert each row to document
        for idx, row in df.iterrows():
            row_content = f"Record {idx + 1}:\n"
            for col in df.columns:
                row_content += f"- {col}: {row[col]}\n"
            doc = Document(
                page_content=row_content,
                metadata={"source": file_path, "row_index": idx}
            )
            docs.append(doc)

        # Add dataset summary
        summary_content = f"Dataset Summary for {os.path.basename(file_path)}:\n"
        summary_content += f"Total records: {len(df)}\n"
        summary_content += f"Columns: {', '.join(df.columns.tolist())}\n\n"
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

def load_and_split_documents(documents_dir: str) -> List[Document]:
    all_docs = []
    
    report_docs_path = os.path.join(documents_dir, "MREPORT.csv")
    if os.path.exists(report_docs_path):
        logger.info(f"Loading file: {report_docs_path}")
        csv_docs = load_csv_as_document(report_docs_path)
        all_docs.extend(csv_docs)
        logger.info(f"Loaded {len(csv_docs)} documents from CSV: {report_docs_path}")
    else:
        logger.warning(f"{report_docs_path} not found.")

    return all_docs

# Load documents
all_docs = load_and_split_documents(DOCUMENTS_DIR)
text_chunks, retriever = None, None

if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    text_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Loaded {len(all_docs)} docs, split into {len(text_chunks)} chunks.")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.warning("Using deprecated HuggingFaceEmbeddings.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    logger.info("FAISS retriever ready.")
else:
    logger.warning("No documents loaded. RAG not available.")

# Updated prompt for Report Data chatbot
prompt_template = """
[ROLE]
You are an expert Report Data assistant for GoodBooks Technologies.

[TASK]
Answer user questions strictly from the report data (CSV or other uploaded reports).
Do not use outside/pretrained knowledge.

[CONTEXT]
Report Data context:
{context}

[CONVERSATION HISTORY]
{history}

[REASONING]
- Analyze the given report data carefully.
- If the answer exists, summarize clearly.
- If missing, do not invent.

[OUTPUT]
Provide a clear and professional answer.

[CONDITIONS]
- If the context does not contain the answer, reply only with:
  "I don't know. Please try asking a different report-related question."
- Never hallucinate.

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
        # | llm
        | StrOutputParser()
    )

@app.post("/gbaiapi/Report-chat", tags=["Goodbooks Ai Api"])
async def report_chat(message: Message, Login: str = Header(...)):
    user_input = message.content.strip()

    # Parse login header
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})

    user_input = spell_check(user_input)

    greetings = ["hi","hello","hey","good morning","good afternoon","good evening","howdy","greetings","what's up","sup"]
    if any(g in user_input.lower() for g in greetings):
        formatted_answer = "Hello! I’m your Report Data assistant. Ask me anything about the uploaded report data."
        return {"response": formatted_answer}

    try:
        history_str = ""

        if retriever:
            answer = chain.invoke({"question": user_input, "history": history_str})
        else:
            fallback_prompt = f"Only answer based on data context. Human: {user_input}\nAssistant:"
            answer = llm.invoke(fallback_prompt).content

        cleaned_answer = clean_response(answer)
        formatted_answer = format_as_points(cleaned_answer)


        return {"response": formatted_answer}

    except Exception as e:
        logger.error(f"Chat error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"response": "Error while processing your request. Please try again."}
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
    uvicorn.run(app, host="0.0.0.0", port=8082)
