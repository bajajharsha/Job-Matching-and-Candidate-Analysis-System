from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import logging
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from dotenv import load_dotenv  # type: ignore
from io import BytesIO
from PyPDF2 import PdfReader

load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

app = FastAPI(
    title="Job Matching API",
    version="1.0",
    description="API for receiving job applications and descriptions."
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.post("/upload/")
async def upload_files(resume: UploadFile = File(...)):
    # file_content = await resume.read()
    
    # temp_file_path = f"/tmp/{resume.filename}"
    # with open(temp_file_path, "wb") as temp_file:
    #     temp_file.write(await resume.read())
    
    resume_content = await resume.read()
    
    with open("uploaded_resume.txt", "wb") as resume_file:  # Writing in binary mode
        resume_file.write(resume_content)
        
    resume_text = resume_content.decode('utf-8')
        
    # Load the PDF content directly from the file content
    pdf_loader = PyPDFLoader(resume_text)
    text_documents_pdf = pdf_loader.load()
    return JSONResponse(content={text_documents_pdf})
    
    # pdf_reader = PdfReader(BytesIO(file_content))
    # text_documents_pdf = []
    # for page_num in range(pdf_reader.()):
    #     page = pdf_reader.getPage(page_num)
    #     page = len(pdf_reader.page_num)
    #     text_documents_pdf.append(page.extract_text())
    
    # Return the extracted text as JSON response
    # return JSONResponse(content={"text_documents_pdf": text_documents_pdf})
    # return  {"filename": resume.filename}

if __name__ == "__main__":
    logger.debug("Starting the FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
