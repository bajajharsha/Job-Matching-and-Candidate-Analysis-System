import logging.config
import google.generativeai as genai
import os
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

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