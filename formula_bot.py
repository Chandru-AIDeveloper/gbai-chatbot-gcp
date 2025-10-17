import json
import os
import logging
import traceback
import re
import pandas as pd
# from langchain_ollama import ChatOllama
from typing import List
from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = "data/"

class Message(BaseModel):
    content: str

def spell_check(text: str) -> str:
    return text

def clean_response(text: str) -> str:
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return '\n'.join(cleaned_lines)


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

def load_csv_as_document(file_path: str) -> Document:
    try:
        df = pd.read_csv(file_path)
        # Convert each formula row into natural language description for better retrieval
        contents = []
        for idx, row in df.iterrows():
            formula_name = row.get("formula_name", f"Formula {idx+1}")
            formula_expression = row.get("formula_expression", "")
            description = row.get("description", "")
            content = (f"Formula Name: {formula_name}\n"
                       f"Formula Expression: {formula_expression}\n"
                       f"Description: {description}\n")
            contents.append(content)
        full_content = "\n\n".join(contents)
        return Document(
            page_content=full_content,
            metadata={"source": file_path, "type": "csv", "rows": len(df)}
        )
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return None

def load_and_split_documents(documents_dir: str) -> List[Document]:
    all_docs = []
    
    formula_file_path = os.path.join(documents_dir, "MFORMULAFIELD.csv")
    if os.path.exists(formula_file_path):
        logger.info(f"Loading file: {formula_file_path}")
        doc = load_csv_as_document(formula_file_path)
        if doc:
            all_docs.append(doc)
        logger.info(f"Loaded documents from CSV: {formula_file_path}")
    else:
        logger.warning(f"{formula_file_path} not found.")

    return all_docs

# Load documents at startup
all_docs = load_and_split_documents(DOCUMENTS_DIR)
text_chunks = None
retriever = None

if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    text_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Loaded {len(all_docs)} docs, split into {len(text_chunks)} chunks.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15}
    )
    logger.info("FAISS vectorstore and retriever initialized.")
else:
    logger.warning("No documents loaded. RAG will not be available.")

prompt_template = """
[ROLE]
You are a specialized Formula assistant for GoodBooks Technologies. You ONLY provide information based on the company's Formula data and documents.

[TASK]
Answer the user's formula-related question using ONLY the information from the provided context. Generate natural, conversational responses similar to ChatGPT style - friendly, helpful, and well-structured.

[CONTEXT]
Here is the relevant Formula data and information:
{context}

[CONVERSATION HISTORY]
Previous conversation:
{history}

[REASONING]
1. Carefully analyze the provided data context above
2. Look for information that directly answers the user's question
3. If the information exists in the context, provide a comprehensive answer
4. Structure your response in a natural, conversational manner
5. Use specific data points, numbers, or details from the context when available

[CONDITIONS]
- CRITICAL: You must ONLY use information from the provided context above
- Do NOT use any pretrained knowledge or external information
- If the context doesn't contain the answer, respond with: "I don't have that information in the Formula system. Could you please ask about something else related to our Formula data?"
- Always be helpful and professional in your tone
- When presenting data from tables, organize it clearly
- Reference specific data points when answering

[OUTPUT FORMAT]
Provide a clear, natural language response that:
- Directly answers the user's question
- Uses a conversational, ChatGPT-like tone
- Includes specific details from the context
- Is well-organized and easy to read

User Question: {question}

Assistant Response:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

if retriever:
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown source')
            formatted.append(f"Source {i} ({source}):\n{content}\n")
        return "\n".join(formatted)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

def extract_json_from_answer(answer_text: str):
    try:
        return json.loads(answer_text)
    except Exception:
        match = re.search(r'(\{[\s\S]+\})', answer_text)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                pass
        return None

def extract_formula_list_to_json(answer_text: str):
    matches = re.findall(r"\d+\.\s([^\n]+)", answer_text)
    if matches:
        formulas = [{"id": i + 1, "name": name.strip()} for i, name in enumerate(matches)]
        return {"formulas": formulas}
    return None

@app.post("/gbaiapi/chat", tags=["Goodbooks Ai Api"])
async def chat(message: Message, Login: str = Header(...)):
    user_input = message.content.strip()
    try:
        login_dto = json.loads(Login)
        username = login_dto.get("UserName", "anonymous")
    except Exception:
        return JSONResponse(status_code=400, content={"response": "Invalid login header"})
    user_input = spell_check(user_input)
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if user_input.lower() in greetings:
        formatted_answer = "Hello! I'm your Formula assistant for GoodBooks Technologies. I can help you with information from our Formula system. What would you like to know about?"
        return {"response": formatted_answer}
    try:
        if not retriever:
            return JSONResponse(
                status_code=503,
                content={"response": "Formula data is not available. Please ensure the data directory contains your Formula files."}
            )
        history_str = ""
        response_data = {
            "question": user_input,
            "history": history_str
        }
        answer = chain.invoke(response_data)
        cleaned_answer = clean_response(answer)

        structured_json = extract_json_from_answer(cleaned_answer)
        if structured_json is not None:
            return structured_json
        else:
            formulas_json = extract_formula_list_to_json(cleaned_answer)
            if formulas_json is not None:
                return formulas_json
            return {"response": cleaned_answer}
    except Exception:
        logger.error(f"Chat error: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"response": "I encountered an error while processing your request. Please try again."},
        )


@app.get("/gbaiapi/system_status", tags=["Goodbooks Ai Api"])
async def get_system_status():
    status = {
        "rag_available": retriever is not None,
        "documents_loaded": len(all_docs) if all_docs else 0,
        "chunks_created": len(text_chunks) if text_chunks else 0,
    }
    return status

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8084)
