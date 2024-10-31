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
import google.generativeai as genai
import json

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

def processUploadedFiles(document_text, document_type):
    logging.info("Processing uploaded files using NLP.")
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

    Please structure the output in a clear pure python nested dictionary format only and extra keywords without the "python" prefix and do not add lists into the format, with each section labeled accordingly. If a section is not present in the document, simply omit it from the python nested dictionary output.

    Here is the {document_type} text to analyze:
    {document_text}
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    # logging.info("Generated output: %s", response.text)
    return response.text

def initialize_pinecone():
    """Initializes the Pinecone client and ensures the index exists."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

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

def store_in_pinecone(document, document_type):
    """Store sections in Pinecone for both job descriptions and resumes."""
    
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    index = initialize_pinecone()  # Ensure the index is initialized
    logging.info("Created Pinecone index for document matching.")
    embeddings_data = []
    logging.info("resume sections: " + str(document))
    cleaned_string = document[9:-3]

    # Check if the document is a JD or a resume
    if document_type.lower() == "job description":
        # Assuming document is in JSON format similar to your example
        sections = json.loads(cleaned_string)  # Load JSON from the document

        # Loop through each JD section
        for jd_name, jd_sections in sections.items():
            logging.info(f"Processing Job Description: {jd_name}")
            for section_key, section_value in jd_sections.items():
                # logging.info(f"Processing section: {section_key}")
                # logging.info(f"Section content: {section_value}")
                
                # Create a unique ID for each section
                section_id = f"jd-{jd_name.lower().replace(' ', '-')}-{section_key.lower().replace(' ', '-')}-{str(uuid4())}"
                
                # Convert the section to JSON string for storage
                section_content = json.dumps({section_key: section_value})

                # Generate embeddings for the section content
                embedding = model.encode(section_content).tolist()

                # Prepare data for upsert
                embeddings_data.append((section_id, embedding, {
                    "content": section_content,
                    "document_type": document_type,
                    "section_name": section_key,
                    "doc_name": jd_name
                }))

    elif document_type.lower() == "resume":
        # Assuming the resume is structured similarly to the JD
        sections = json.loads(cleaned_string)  # Load JSON from the document

        # Loop through each section of the resume
        logging.info(f"Processing Resume")
        for section_key, section_value in sections.items():
            logging.info(f"Processing section: {section_key}")
            logging.info(f"Section content: {section_value}")

            # Create a unique ID for each section
            section_id = f"resume-{section_key.lower().replace(' ', '-')}-{str(uuid4())}"
            
            # Convert the section to JSON string for storage
            section_content = json.dumps({section_key: section_value})

            # Generate embeddings for the section content
            embedding = model.encode(section_content).tolist()

            # Prepare data for upsert
            embeddings_data.append((section_id, embedding, {
                "content": section_content,
                "document_type": document_type,
                "section_name": section_key
            }))

    # Upsert the vectors to the Pinecone index
    index.upsert(vectors=embeddings_data)
    logging.info(f"Stored {len(embeddings_data)} sections in Pinecone for {document_type}.")

def find_best_job_match(resume_sections):
    """Find the best job match for a given resume."""
    index = initialize_pinecone()  # Ensure the index is initialized
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    cleaned_string = resume_sections[9:-3]
    logging.info("resume sections: " + str(cleaned_string))
    sections = eval(cleaned_string)  # Load JSON from the document
    
    final_result = {}
    
    # Loop through each section of the resume
    for section_key, section_value in sections.items():
        logging.info(f"Processing section: {section_key}")
        logging.info(f"Section content: {section_value}")

        # Create a unique ID for each section
        section_id = f"resume-{section_key.lower().replace(' ', '-')}-{str(uuid4())}"
        
        # Convert the section to JSON string for storage
        section_content = json.dumps({section_key: section_value})

        # Generate embeddings for the section content
        query_embedding = model.encode(section_content).tolist()
        result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        # Store results in final_result with section ID as the key
        final_result[section_id] = {
            "section": section_key,
            "matches": result['matches']  # Store the top matching results for the section
        }
        
        # logging.info("final result: " + str(final_result))        
        
    return final_result

def finalLLM(context):
    logging.info("Final processing for output")
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))
    
    prompt = f"""
    You are an advanced AI model specialized in evaluating the alignment between a candidate’s resume and a job description. 
    Using the provided context below, generate a detailed, frontend-ready analysis to show why a candidate is a good fit for the role. 
    The output should be structured with easy-to-read headings and descriptions in each category (Skill Match, Experience Match, 
    Education Fit, and Technological Fit).

    The format should be in structured JSON for direct frontend display, with clear labels and explanations for each section:

    1. **Skill Match**: Provide a percentage indicating the overlap between required skills in the job description and those 
       in the candidate’s resume. Include a list of matched skills and missing skills, with labels to make it readable on a 
       frontend display.
    2. **Experience Match**: 
       - Calculate and present the overall relevance percentage of the candidate's experience to the job’s requirements.
       - List each relevant job, showing the job title, relevance score, and a brief explanation of the overlap in responsibilities.
       - Include a list of job responsibilities that match those required by the role.
    3. **Education Fit**: 
       - Provide a match percentage for education, indicating how well the candidate’s educational background aligns with 
         the job’s requirements.
       - List matched and missing qualifications with brief descriptions where necessary.
    4. **Technological Fit**: 
       - Present a percentage score indicating the overlap between required and possessed technologies.
       - List matched technologies and missing technologies, if any.

    Output the information in the following structured JSON format for direct frontend display:

    {{
        "Skill Match": {{
            "matchPercentage": "X%",
            "description": "Percentage of required skills possessed by the candidate.",
            "matchedSkills": {{
                "label": "Matched Skills",
                "skills": ["Skill 1", "Skill 2", "..."]
            }},
            "missingSkills": {{
                "label": "Missing Skills",
                "skills": ["Skill 3", "Skill 4", "..."]
            }}
        }},
        "Experience Match": {{
            "overallRelevance": {{
                "percentage": "X%",
                "description": "Overall relevance of the candidate's experience to the job."
            }},
            "relevantExperience": {{
                "label": "Relevant Past Experience",
                "jobs": [
                    {{
                        "jobTitle": "Title 1",
                        "relevance": "X%",
                        "details": "Explanation of relevance."
                    }},
                    {{
                        "jobTitle": "Title 2",
                        "relevance": "Y%",
                        "details": "Explanation of relevance."
                    }}
                ]
            }},
            "responsibilityOverlap": {{
                "label": "Matched Responsibilities",
                "responsibilities": ["Responsibility 1", "Responsibility 2", "..."]
            }}
        }},
        "Education Fit": {{
            "matchPercentage": "X%",
            "description": "Percentage indicating how well the candidate’s education aligns with the job requirements.",
            "matchedQualifications": {{
                "label": "Matched Qualifications",
                "qualifications": ["Qualification 1", "Qualification 2", "..."]
            }},
            "missingQualifications": {{
                "label": "Missing Qualifications",
                "qualifications": ["Qualification 3"]
            }}
        }},
        "Technological Fit": {{
            "matchPercentage": "X%",
            "description": "Percentage of required technologies that the candidate is familiar with.",
            "matchedTechnologies": {{
                "label": "Matched Technologies",
                "technologies": ["Technology 1", "Technology 2", "..."]
            }},
            "missingTechnologies": {{
                "label": "Missing Technologies",
                "technologies": ["Technology 3"]
            }}
        }},
        "BestJobMatch": {{
            The bestt job match based on the analysis
        }}
    }}
    
    FInally give a detailed and on point note on which job description the resume is best suited for based on the analysis in the json itself.
    The final ouptput should be in JSON format only and no extra keywords should be added to the format. 

    Context: {context}
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    logging.info("Generated final output: %s", response.text)
    return response.text

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
        # logging.info("Job description sections: " + str(jd_sections))
    
    # Store job description sections in Pinecone
    store_in_pinecone(jd_sections, document_type="job description")
    
    resume_text = " ".join(doc.page_content for doc in resume_documents)
    resume_sections = processUploadedFiles(resume_text, document_type="resume")
    # logging.info("Resume sections: " + str(resume_sections))
    
    # Store resume sections in Pinecone
    # store_in_pinecone(resume_sections, document_type="resume")
    
    # Perform query to find the best job match
    context= find_best_job_match(resume_sections)
    
    # call final llm
    final_output = finalLLM(context)
    
    return JSONResponse(content={"detail": final_output[7:-3]})    

    


if __name__ == "__main__":
    uvicorn.run(app,
        host="0.0.0.0",
        port=8002)

