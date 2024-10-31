import google.generativeai as genai
import logging
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader  # type: ignore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

def genAI(document_text, document_type): 
    """Generate a structured output for resumes or job descriptions using Google Gemini."""
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

    prompt = f"""
    You are a highly advanced language model capable of understanding and analyzing {document_type}. Your task is to process the following plain text {document_type} and identify the different sections, categorizing them accordingly. The sections we are interested in are:

    1. **Personal Information**: For resumes only—Name, contact details, and any relevant links.
    2. **Summary or Objective**: A brief overview of the candidate’s career goals (for resumes) or the role’s objectives (for JDs).
    3. **Skills**: A list of relevant skills, including technical and soft skills.
    4. **Experience**: Work history including job titles, companies, duration of employment (for resumes) or required experience (for JDs).
    5. **Education**: Academic qualifications including degrees, institutions, and years of graduation (for resumes) or required qualifications (for JDs).
    6. **Certifications**: Any relevant certifications (for resumes) or preferred qualifications (for JDs).
    7. **Responsibilities**: Specific job responsibilities (for JDs only).

    Please structure the output in a clear JSON format, with each section labeled accordingly. If a section is not present in the document, simply omit it from the JSON output.

    Here is the {document_type} text to analyze:
    {document_text}
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    logging.info("Generated output: %s", response.text)
    print(response.text)

# Load and analyze resume
pdf_loader = PyPDFLoader("r2.pdf")
resume_documents = pdf_loader.load()
resume_text = " ".join([doc.page_content for doc in resume_documents])  # Combine pages into a single string
genAI(resume_text, document_type="resume")

# Load and analyze job description
txt_loader = TextLoader("jds.txt")
jd_documents = txt_loader.load()
jd_text = " ".join([doc.page_content for doc in jd_documents])  # Combine pages into a single string
genAI(jd_text, document_type="job description")
