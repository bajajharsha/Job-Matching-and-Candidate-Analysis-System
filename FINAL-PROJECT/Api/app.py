import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
from dotenv import load_dotenv  # type: ignore
from pathlib import Path
from typing import List
import logging.config
import json
from pinecone import Pinecone, ServerlessSpec  # type: ignore
from textExtraction import extract_from_pdf, extract_from_docx, extract_from_txt, extract_from_json
from nlp import processUploadedFiles
from vectorStore import store_in_pinecone
from similaritySearch import find_best_job_match
from finalLLM import finalLLM
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

app = FastAPI(
    title="Job Matching API",
    version="1.0",
    description="API for receiving job applications and descriptions."
)

# Create directory for uploads
UPLOAD_DIR = Path("/home/harsha/Desktop/projectuploadedFiles")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def delete_index_if_exists(index_name):
    """Delete the index if it exists."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)
        logging.info(f"Deleted index: {index_name}")

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
 
    # Extract from Job Descriptions
    all_jd_documents = []
    for jd in jds:
        if jd.content_type == 'text/plain':
            jd_documents = extract_from_txt(jd)
        elif jd.content_type == 'application/json':
            jd_documents = extract_from_json(jd)
        else:
            raise HTTPException(status_code=400, detail="Unsupported job description file type.")
        
        all_jd_documents.extend(jd_documents)  # Extend to combine text from all job descriptions
        
    jd_texts = " ".join(doc.page_content for doc in all_jd_documents)
    jd_sections = processUploadedFiles(jd_texts, document_type="job description")

    # Store job description sections in Pinecone
    store_in_pinecone(jd_sections, document_type="job description")
    
    resume_text = " ".join(doc.page_content for doc in resume_documents)
    resume_sections = processUploadedFiles(resume_text, document_type="resume")
    
    # Perform query to find the best job match
    context = find_best_job_match(resume_sections)
    
    # Call final llm
    final_output = finalLLM(context)
    final_result = json.loads(final_output[7:-3])
    
    # Delete pinecone index after processing the request 
    delete_index_if_exists("job-matching")
    # os.rmdir(file_location)
    # shutil.rmtree(file_location)
    
    return final_result
    


if __name__ == "__main__":
    uvicorn.run(app,
        host="0.0.0.0",
        port=8001)

