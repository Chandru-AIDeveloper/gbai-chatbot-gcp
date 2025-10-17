import json
import os
import logging
import re
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import Header
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DOCUMENTS_DIR = "data/GBDBSCRIPTS"
SCHEMA_CHAT_HISTORY_DB = "schema_chat_history.db"

class SchemaTable(BaseModel):
    table_name: str
    columns: List[Dict[str, str]]
    constraints: List[str]
    foreign_keys: List[str]
    indexes: List[str]

class SchemaParser:
    @staticmethod
    def parse_sql_file(file_path: str) -> List[SchemaTable]:
        """Parse SQL DDL file and extract table schema information"""
        tables = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Find CREATE TABLE statements
            create_table_pattern = r'CREATE\s+TABLE\s+(?:\[?[A-Za-z_][A-Za-z0-9_]*\]?\.)?\[?([A-Za-z_][A-Za-z0-9_]*)\]?\s*\((.*?)\)(?:\s*GO)?'
            
            matches = re.finditer(create_table_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                table_name = match.group(1)
                table_body = match.group(2)
                
                table_info = SchemaParser._parse_table_body(table_name, table_body)
                tables.append(table_info)
                
        except Exception as e:
            logger.error(f"Error parsing SQL file {file_path}: {e}")
        
        return tables
    
    @staticmethod
    def _parse_table_body(table_name: str, table_body: str) -> SchemaTable:
        """Parse the body of a CREATE TABLE statement"""
        columns = []
        constraints = []
        foreign_keys = []
        indexes = []
        
        # Split by lines and clean up
        lines = [line.strip() for line in table_body.split('\n') if line.strip()]
        
        for line in lines:
            line = line.rstrip(',')
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if it's a constraint
            if line.upper().startswith('CONSTRAINT'):
                if 'FOREIGN KEY' in line.upper():
                    foreign_keys.append(line)
                else:
                    constraints.append(line)
            elif line.upper().startswith('PRIMARY KEY') or line.upper().startswith('UNIQUE'):
                constraints.append(line)
            elif line.upper().startswith('CHECK'):
                constraints.append(line)
            else:
                # It's likely a column definition
                column_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z0-9_()]+)', line, re.IGNORECASE)
                if column_match:
                    column_name = column_match.group(1)
                    column_type = column_match.group(2)
                    
                    # Extract additional column properties
                    properties = []
                    if 'NOT NULL' in line.upper():
                        properties.append('NOT NULL')
                    if 'PRIMARY KEY' in line.upper():
                        properties.append('PRIMARY KEY')
                    if 'UNIQUE' in line.upper():
                        properties.append('UNIQUE')
                    if 'DEFAULT' in line.upper():
                        default_match = re.search(r'DEFAULT\s+([^,\s]+)', line, re.IGNORECASE)
                        if default_match:
                            properties.append(f'DEFAULT {default_match.group(1)}')
                    
                    columns.append({
                        'name': column_name,
                        'type': column_type,
                        'properties': ', '.join(properties) if properties else ''
                    })
        
        return SchemaTable(
            table_name=table_name,
            columns=columns,
            constraints=constraints,
            foreign_keys=foreign_keys,
            indexes=indexes
        )

def load_schema_documents(documents_dir: str) -> List[Document]:
    """Load all schema files from the data directory and subdirectories"""
    all_docs = []
    
    if not os.path.exists(documents_dir):
        logger.warning(f"Documents directory {documents_dir} does not exist")
        return all_docs
    
    # Find all SQL and TXT files recursively in subdirectories
    schema_files = []
    for root, dirs, files in os.walk(documents_dir):
        for file in files:
            if file.lower().endswith(('.sql', '.txt')):
                schema_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(schema_files)} schema files: {schema_files}")
    
    for file_path in schema_files:
        try:
            tables = SchemaParser.parse_sql_file(file_path)
            
            for table in tables:
                # Create comprehensive document content for each table
                content = f"Table: {table.table_name}\n\n"
                
                content += "Columns:\n"
                for col in table.columns:
                    content += f"- {col['name']}: {col['type']}"
                    if col['properties']:
                        content += f" ({col['properties']})"
                    content += "\n"
                
                if table.constraints:
                    content += "\nConstraints:\n"
                    for constraint in table.constraints:
                        content += f"- {constraint}\n"
                
                if table.foreign_keys:
                    content += "\nForeign Keys:\n"
                    for fk in table.foreign_keys:
                        content += f"- {fk}\n"
                
                # Create document
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "table_name": table.table_name,
                        "type": "table_schema"
                    }
                )
                all_docs.append(doc)
                
                # Create relationship documents for foreign keys
                for fk in table.foreign_keys:
                    if 'REFERENCES' in fk.upper():
                        ref_match = re.search(r'REFERENCES\s+([A-Za-z_][A-Za-z0-9_]*)', fk, re.IGNORECASE)
                        if ref_match:
                            referenced_table = ref_match.group(1)
                            relationship_content = f"Relationship: {table.table_name} -> {referenced_table}\n"
                            relationship_content += f"Foreign Key: {fk}\n"
                            
                            rel_doc = Document(
                                page_content=relationship_content,
                                metadata={
                                    "source": file_path,
                                    "from_table": table.table_name,
                                    "to_table": referenced_table,
                                    "type": "relationship"
                                }
                            )
                            all_docs.append(rel_doc)
            
            logger.info(f"Loaded {len(tables)} tables from {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    return all_docs

# Initialize components globally
logger.info(f"Loading schema documents from directory: {DOCUMENTS_DIR}")

all_docs = load_schema_documents(DOCUMENTS_DIR)
text_chunks, retriever = None, None

if all_docs:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Loaded {len(all_docs)} schema documents, split into {len(text_chunks)} chunks.")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    logger.info("FAISS retriever ready for schema queries.")
else:
    logger.warning("No schema documents loaded. Schema chat not available.")

# Initialize LLM
llm = ChatOllama(
    model="gemma:2b",
    base_url="http://ollama:11434",
    temperature=0.3
)
# ChatGPT-style conversational prompt
prompt_template = """
You are GoodBooks Database Schema Assistant, a friendly and conversational database schema expert.

IMPORTANT: You are having a natural conversation with the user about GoodBooks database schema. Don't dump raw data or list everything - instead, have a helpful conversation and answer their specific questions naturally.

Available Database Schema Context:
{context}

CONVERSATION GUIDELINES:
1. Be conversational and natural like ChatGPT
2. Answer specific questions, don't list everything unless asked
3. If user asks "What tables do we have?" - give a conversational overview, not a data dump
4. If user asks about a specific table - explain it naturally with key points
5. Use the schema data to inform your answers but respond conversationally
6. Be helpful and engaging, not robotic also dont answer questions outside your knowledge and pretrained data use only the context data provided.
7. Focus only on GoodBooks database schema and related topics

EXAMPLES:
User: "What tables do we have?"
Good: "I can see you have several HR-related tables in your GoodBooks schema! There are tables for employee appraisals, holiday management, pay structures, performance metrics, and more. Would you like me to tell you about any specific table?"

User: "Tell me about APPRAISAL"
Good: "The APPRAISAL table handles employee performance appraisals. It stores appraisal records with details like appraisal IDs, employee references, and performance scores. It connects to other HR tables through foreign key relationships. What specific aspect would you like to know more about?"

User: "{question}"

Response (be conversational and natural about GoodBooks database schema from the given script data dont answer from pretrained data and outside of given context data):"""

prompt = ChatPromptTemplate.from_template(prompt_template)

if retriever:
    def format_context(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    chain = (
        {
            "context": lambda x: format_context(retriever.invoke(x["question"])),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# Main chat function for orchestration integration
async def chat(message, Login: str = None):
    """Main chat function for orchestration integration"""
    try:
        user_input = message.content.strip() if hasattr(message, 'content') else str(message).strip()
        
        # Parse login header if provided
        username = "orchestrator"  # default
        if Login:
            try:
                login_dto = json.loads(Login)
                username = login_dto.get("UserName", "orchestrator")
            except:
                pass
        
        # Handle simple greetings
        simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if user_input.lower().strip() in simple_greetings:
            response = "Hello! I'm your GoodBooks Database Schema Assistant. I can help you understand table structures, relationships, constraints, and column definitions in your GoodBooks database. What would you like to know about the schema?"
            return {"response": response}
        
        # Process the question using the chain
        if retriever:
            answer = chain.invoke({"question": user_input})
        else:
            # Fallback response if no retriever available
            fallback_prompt = f"""
You are GoodBooks Database Schema Assistant, specialized in database schema analysis. 
You help users understand GoodBooks database structures, tables, relationships, and constraints.

User: {user_input}
Assistant:"""
            answer = llm.invoke(fallback_prompt).content
        
        # Clean and format response
        cleaned_answer = answer.strip()
        
        return {"response": cleaned_answer}
        
    except Exception as e:
        logger.error(f"Schema bot error: {e}")
        
        # Check if it's an Ollama connection error
        if "ConnectError" in str(e) or "Connection" in str(e):
            error_response = "I'm having trouble connecting to the AI service. Please make sure Ollama is running with 'ollama serve' command, then try again."
        else:
            error_response = "I apologize, but I encountered an error processing your database schema question. Please try again, and I'll do my best to help you understand the GoodBooks database structure."
        
        return {"response": error_response}

# Additional utility functions
def get_available_tables():
    """Get list of available tables"""
    if not all_docs:
        return []
    
    tables = []
    for doc in all_docs:
        if doc.metadata.get("type") == "table_schema":
            tables.append({
                "table_name": doc.metadata.get("table_name"),
                "source_file": os.path.basename(doc.metadata.get("source", ""))
            })
    
    return tables

def is_schema_bot_available():
    """Check if schema bot is properly initialized"""
    return retriever is not None and len(all_docs) > 0

# Log initialization status
logger.info(f"Schema bot initialized - Available: {is_schema_bot_available()}, Tables loaded: {len(get_available_tables())}")