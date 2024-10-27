import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, JSONLoader  # type: ignore
from dotenv import load_dotenv  # type: ignore
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from typing import List
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
import logging
import json
import logging.config
from langchain_community.vectorstores import Chroma
from uuid import uuid4



# Load logging configuration
with open("FINAL/API/logging_config.json", "r") as file:
    config = json.load(file)
    logging.config.dictConfig(config)
    
# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

app = FastAPI(
    title="Job Matching API",
    version="1.0",
    description="API for receiving job applications and descriptions."
)

# Create directory for uploads
UPLOAD_DIR = Path.home() / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_file(file: UploadFile) -> Path:
    """Saves an uploaded file to the designated directory and returns its path."""
    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as file_object:
        file_object.write(file.file.read())
    return file_location

def extract_from_pdf(resume: UploadFile):
    logging.info("Extracting text from PDF.")
    file_location = save_file(resume)
    pdf_loader = PyPDFLoader(file_location)
    resume_documents = pdf_loader.load()
    return resume_documents
def extract_from_docx(resume: UploadFile):
    file_location = save_file(resume)
    docx_loader = Docx2txtLoader(file_location)
    resume_documents = docx_loader.load()
    return resume_documents

def extract_from_txt(resume: UploadFile):
    file_location = save_file(resume)
    txt_loader = TextLoader(file_location)
    resume_documents = txt_loader.load()
    return resume_documents
def extract_from_json(resume: UploadFile):
    logging.info("Extracting text from JSON.")
    file_location = save_file(resume)
    json_loader = JSONLoader(
        file_path=file_location,
        jq_schema=".",
        text_content = False
        )
    resume_documents = json_loader.load()
    return resume_documents

def chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents= text_splitter.split_documents(documents)
    return documents

def add_to_vector_store(embeddings, chunked_resume_documents):
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "project"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        
    index = pc.Index(index_name)
    
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    uuids = [str(uuid4()) for _ in range(len(chunked_resume_documents))]

    vector_store.add_documents(documents=chunked_resume_documents, ids=uuids)



def create_embedding(document_texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(document_texts, convert_to_tensor=True)
    logging.info("Type of embeddings: " + str(type(embeddings)))
    return embeddings  

@app.post("/upload/")
async def upload_files(resume: UploadFile = File(...), jds: List[UploadFile] = File(...)):
    logging.info("Upload the files.")
    response_content = {
        "resume_documents": None,
        "jd_documents": None,
        "resume_embeddings": [], 
        "jd_embeddings": []  
    }

    # Extract from resume
    if resume.content_type == 'application/pdf':
        resume_text = extract_from_pdf(resume)
    elif resume.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        resume_text = extract_from_docx(resume)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")
    
    # logging.info("Extracted text from resume." + json.dumps(resume_text))
    
    # Chunk the resume text
    chunked_resume_documents = chunk_text(resume_text)
    # document_texts = [doc.page_content for doc in chunked_resume_documents] 
    document_texts = ["This is an example sentence", "Esta es una sentencia de ejemplo"]
    logging.info("Type of document_texts: " + str(type(document_texts)))
    # logging.info("Type of document_texts: " + document_texts)
    # chunked_resume_serializable = [{"page_content": doc.page_content} for doc in chunked_resume_documents]
    # logging.info("Chunked resume text: " + json.dumps(chunked_resume_serializable))

    # Create embeddings for each chunk of the resume
    # for chunk in chunked_resume_documents:
    embedding = create_embedding(document_texts)  # Use the chunk's text for embedding
    add_to_vector_store(embedding, chunked_resume_documents)
    
    # Extract from job descriptions
    all_jd_texts = []
    for jd in jds:
        if jd.content_type == 'text/plain':
            jd_text = extract_from_txt(jd)
        elif jd.content_type == 'application/json':
            jd_text = extract_from_json(jd)
        else:
            raise HTTPException(status_code=400, detail="Unsupported job description file type.")
            
    all_jd_texts.extend(jd_text)  # Extend to combine text from all job descriptions
    # logging.info("Extracted text from job descriptions." + json.dumps(all_jd_texts))
    return JSONResponse(content=response_content)

    
    # response_content["jd_documents"] = " ".join(all_jd_texts)
    
    # Chunk the combined JDs
    # chunked_jd_documents = chunk_text(response_content["jd_documents"].split())

    # # Create embeddings for each chunk of JDs
    # for chunk in chunked_jd_documents:
    #     embedding = create_embedding(chunk)  # Use the chunk's text for embedding
    #     response_content["jd_embeddings"].append(embedding)
        
    # # Add documents and embeddings to the vector store
    # add_to_vector_store(response_content["resume_embeddings"], chunked_resume_documents, uuids)

    # return JSONResponse(content=response_content)

if __name__ == "__main__":
    uvicorn.run(app,
        host="0.0.0.0",
        port=8001)