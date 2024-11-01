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

file_location = UPLOAD_DIR

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
    
    # Define the prompts for resumes and job descriptions
    if document_type == "resume":
        prompt = f"""
        You are a highly advanced language model capable of understanding and analyzing resumes. Your task is to process the following plain text resume and identify the different sections, categorizing them accordingly and in detailed using all the available text in the resume. The sections we are interested in are:

        1. **Personal Information**: For resumes only—Name, contact details, and any relevant links.
        2. **Summary or Objective**: A brief overview of the candidate’s career goals.
        3. **Skills**: A list of relevant skills, including technical and soft skills and the responsibilties.
        4. **Experience**: Work history including job titles, companies, duration of employment.
        5. **Education**: Academic qualifications including degrees, institutions, and years of graduation.
        6. **Certifications**: Any relevant certifications.
        
        Please structure the output in a clear pure python nested dictionary format only, with each section labeled accordingly. If a section is not present in the document, simply omit it from the output.

        Here is the resume text to analyze:
        {document_text}
        """
        
    elif document_type == "job description":
        prompt = f"""
        You are an advanced language model tasked with understanding and categorizing sections in job descriptions. Your goal is to analyze the following plain text job description and output a structured JSON object that captures the essential sections of the job description.

        Please format the output in the following JSON structure:

        {{
            "Job Description X": {{
                "Position": "Position title",
                "Location": "Location of the job",
                "Job Type": "Type of employment, e.g., Full-Time, Part-Time, Volunteer",
                "Company": "Company name",
                "Summary": "Brief overview of the company and the role",
                "Skills": {{
                    "Technical Skills": "List of relevant technical skills",
                    "Soft Skills": "List of relevant soft skills"
                }},
                "Experience": {{
                    "Required": "Details about required experience"
                }},
                "Education": {{
                    "Required": "Required academic qualifications"
                }},
                "Certifications": {{
                    "Preferred": "Preferred certifications or qualifications"
                }},
                "Responsibilities": "List of primary responsibilities"
            }}
        }}

        If any section is not present in the document, simply omit it from the JSON output.

        Here is the job description text to analyze:
        {document_text}
        """

    # Generate content using the defined prompt
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
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
    cleaned_string = document[7:-3]
    logging.info("cleaned string for adding to db")
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
    Based on the context provided below, generate a detailed analysis demonstrating why a candidate is a suitable fit for the role. 
    Start by identifying the best job fit for the candidate, followed by a comprehensive analysis supporting this conclusion. 
    Structure your output with clear headings and descriptions for each category: Best Job Match, Skill Match, Experience Match, 
    Education Fit, and Technological Fit.

    The output should be formatted in structured JSON for direct frontend integration, ensuring clear labels and explanations for each section:

    **Best Job Match**: 
    - Identify the job description that best aligns with the candidate's profile. 
    - Provide a detailed explanation, including analytics on how the candidate’s skills, experience, and qualifications align with the job requirements. 
    - Highlight key reasons supporting this match, emphasizing any standout qualifications or experiences that make this job the ideal fit.

    1. **Skill Match**: 
       - Indicate the percentage of overlap between the required skills in the job description and those in the candidate’s resume. 
       - Include a list of matched skills and missing skills, labeled for readability on the frontend.
       
    2. **Experience Match**: 
       - Calculate and present the overall relevance percentage of the candidate's experience to the job’s requirements.
       - For each relevant job, include the job title, relevance score, and a brief explanation of the overlap in responsibilities.
       - List job responsibilities that match those required by the role.

    3. **Education Fit**: 
       - Provide a percentage for education match, indicating how well the candidate’s educational background aligns with the job’s requirements.
       - List matched and missing qualifications, providing brief descriptions where necessary.

    4. **Technological Fit**: 
       - Present a percentage score for the overlap between required and possessed technologies.
       - Include lists of matched and missing technologies.

    Output the information in the following structured JSON format for direct frontend display:

    {{
        "BestJobMatch": {{
            "jobDescription": "The job description the resume aligns with best.",
            "reason": "A detailed explanation of why this job is the best fit based on the analysis.",
            "analytics": {{
                "skillsAnalysis": "Analysis of how the candidate's skills match the job requirements.",
                "experienceAnalysis": "Analysis of how the candidate's experience aligns with job responsibilities.",
                "educationAnalysis": "Analysis of how the candidate's education supports the job requirements.",
                "technologyAnalysis": "Analysis of how the candidate's technological expertise fits the job needs."
            }}
        }},
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
        }}
    }}
    
    Ensure all sections are detailed, particularly the Best Job Match section, which should include a thorough explanation of why this job is the best fit for the candidate based on the provided analysis. The final output should be strictly in JSON format without any additional keywords or text. 

    Context: {context}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    logging.info("Generated final output: %s", response.text)
    
    return response.text

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
        port=8002)

