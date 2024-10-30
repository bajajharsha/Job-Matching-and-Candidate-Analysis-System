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
from uuid import uuid4
import json
import logging.config
from langchain_core.documents import Document  # Assuming you have this import from langchain_core
import openai

# Load logging configuration
with open("FINAL/API/logging_config.json", "r") as file:
    config = json.load(file)
    logging.config.dictConfig(config)

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure to set your OpenAI API key

app = FastAPI(
    title="Job Matching API",
    version="1.0",
    description="API for receiving job applications and descriptions."
)

# Create directory for uploads
UPLOAD_DIR = Path.home() / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

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
        text_content=False
    )
    resume_documents = json_loader.load()
    return resume_documents

def chunk_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)
    return documents

def create_embedding(document_texts):
    embeddings = model.encode(document_texts)
    logging.info("Type of embeddings: " + str(type(embeddings)))
    return embeddings.tolist()

def initialize_pinecone():
    """Initializes the Pinecone client and ensures the index exists."""
    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if it does not exist
    if 'job-matching' not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name="job-matching",
            dimension=384,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Replace with the correct region
            )
        )
    return pc.Index("job-matching")  # Removed namespace parameter

def delete_index_if_exists(index_name: str):
    """Delete the index if it exists."""
    pc = Pinecone(api_key=pinecone_api_key)
    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)
        logging.info(f"Deleted index: {index_name}")

def add_to_vector_store(index, text_data):
    """Adds documents to the Pinecone vector store. Embeddings are created here as well"""
    upserted_data = []
    for i, item in enumerate(text_data):
        id = index.describe_index_stats()['total_vector_count']
        upserted_data.append(
            (
                f"doc-{str(id + i)}",  # Create unique IDs without namespace
                model.encode(item).tolist(),
                {
                    'content': item
                }
            )
        )
    index.upsert(vectors=upserted_data)  # Removed namespace parameter
def add_to_vector_store_both(index, resumes, job_descriptions):
    """Adds documents to the Pinecone vector store. Embeddings are created here as well"""
    logging.info("Adding resumes and job descriptions to the vector store.")
    upserted_data = []
    # Prepare upsert data for resumes
    upserted_data = []
    for i, resume in enumerate(resumes):
        embedding = model.encode(resume).tolist()
        upserted_data.append((f"resume-{i}", embedding, {"content": resume}))

    # Prepare upsert data for job descriptions
    for i, job in enumerate(job_descriptions):
        embedding = model.encode(job).tolist()
        upserted_data.append((f"job-{i}", embedding, {"content": job}))
        
    index.upsert(vectors=upserted_data)


def perform_similarity_search(index, query_em):
    """Performs a similarity search in Pinecone."""
    result = index.query(vector=query_em, top_k=5, include_metadata=True)
    logging.info("Result: " + str(result))
    return result

def openAI(result, query):
    system_role = (
 
    "Answer the question as truthfully as possible using the provided context, "
    "and if the answer is not contained within the text and requires some latest information to be updated, "
    "print 'Sorry Not Sufficient context to answer query' \n"

    )

    # Check if there are matches in the result
    if result['matches']:
        context = [match['metadata']['content'] for match in result['matches']]
        context_str = '\n'.join(context) 
        user_input = context_str + '\n' + "What job matches this resume?" + '\n'

        gpt4_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_input}
            ]
        )
        return gpt4_response
    else:
        return "No matches found."


@app.post("/upload/")
async def upload_files(resume: UploadFile = File(...), jds: List[UploadFile] = File(...)):
    
    logging.info("Upload the files.")

    # Extract from resume
    if resume.content_type == 'application/pdf':
        resume_documents = extract_from_pdf(resume)
    elif resume.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        resume_documents = extract_from_docx(resume)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")
    
    
    # Chunk the resume text
    chunked_resume_documents = chunk_text(resume_documents)
    logging.info("Chunked resume text." + str(type(chunked_resume_documents)))
    
    # Prepare text data for embedding and vector store
    resume_texts = [doc.page_content for doc in resume_documents] 
    logging.info("Resume text: " + str(resume_texts))

    # Initialize Pinecone and add resume to vector store
    index = initialize_pinecone()
    logging.info("Pinecone initialized.")
    # add_to_vector_store(resume_index, resume_texts)  # Embeddings are created here
    # return JSONResponse(content={"detail": "Success"})
    # Extract from job descriptions
    all_jd_documents = []
    for jd in jds:
        if jd.content_type == 'text/plain':
            jd_documents = extract_from_txt(jd)
        elif jd.content_type == 'application/json':
            jd_documents = extract_from_json(jd)
        else:
            raise HTTPException(status_code=400, detail="Unsupported job description file type.")
        
        all_jd_documents.extend(jd_documents)  # Extend to combine text from all job descriptions

    # Chunk job descriptions
    chunked_jd_documents = chunk_text(all_jd_documents)

    # Prepare text data for embedding and vector store
    jd_texts = [doc.page_content for doc in all_jd_documents]
    # jd_texts = ""
    logging.info("Job description text: " + str(jd_texts))

    # Add job descriptions to the same Pinecone index
    add_to_vector_store_both(index, resume_texts, jd_texts)  # Store in the same index
    
    # return JSONResponse(content={"detail": "Success"})

    # Perform similarity search
    # query = "Based on the resume what is the current role of the user?"
    query_resume = resume_texts[0]  # Using the first resume as the query
    logging.info("Query resume: " + str(query_resume))
    query_em = model.encode(query_resume).tolist()
    
    result = perform_similarity_search(index,query_em)  # Query from the same index
    
    finalResult = openAI(result,query_resume)
    logging.info("Final Result: " + str(finalResult))
    # return JSONResponse(content={"detail": result})
    

    # Delete indices after processing the request (optional)
    delete_index_if_exists("job-matching")
    
    return finalResult

if __name__ == "__main__":
    uvicorn.run(app,
        host="0.0.0.0",
        port=8002)
