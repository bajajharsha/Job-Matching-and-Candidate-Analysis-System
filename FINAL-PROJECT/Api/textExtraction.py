from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, JSONLoader  # type: ignore
import logging.config
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO)

# Create directory for uploads
UPLOAD_DIR = Path("/home/harsha/Desktop/projectuploadedFiles")
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
        text_content=False
    )
    resume_documents = json_loader.load()
    return resume_documents

