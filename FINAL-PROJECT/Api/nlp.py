import logging.config
import google.generativeai as genai
import os
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

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