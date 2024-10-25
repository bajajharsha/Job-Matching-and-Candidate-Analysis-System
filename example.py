import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import pinecone
from llama_index import Document, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings import OpenAIEmbedding

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")  # Change as needed
index_name = "job-resume-matching"  # Your Pinecone index name

# Initialize LlamaIndex embeddings
embedding_function = OpenAIEmbedding()

# Gemini function
def get_gemini_response(input):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input)
    return response.text

# Convert PDF to text
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Function to classify sections
def classify_sections(text):
    prompt = f"""
    Classify the following text into sections: skills, experience, qualifications, certifications, etc.
    Text: {text}
    Provide the sections in a structured format like this:
    {{
        "skills": "text",
        "experience": "text",
        "qualifications": "text",
        "certifications": "text"
    }}
    """
    response = get_gemini_response(prompt)
    return eval(response)  # Make sure to safely handle the response

# Function to calculate matches
def calculate_matches(resume_sections, jd_sections):
    matches = {}
    
    # Calculate skill match percentage
    required_skills = jd_sections.get("skills", "").split(", ")
    candidate_skills = resume_sections.get("skills", "").split(", ")
    skill_match_count = sum(1 for skill in required_skills if skill in candidate_skills)
    skill_match_percentage = (skill_match_count / len(required_skills)) * 100 if required_skills else 0
    
    matches['Skill Match'] = f"{skill_match_percentage:.2f}%"

    # Experience match
    experience_summary = resume_sections.get("experience", "")
    jd_experience_summary = jd_sections.get("experience", "")
    experience_match = "Match found" if experience_summary and jd_experience_summary in experience_summary else "No direct match"
    
    matches['Experience Match'] = experience_match

    # Education fit
    education_summary = resume_sections.get("qualifications", "")
    jd_education_summary = jd_sections.get("qualifications", "")
    education_fit = "Match found" if jd_education_summary in education_summary else "No direct match"
    
    matches['Education Fit'] = education_fit

    return matches

# Function to get final feedback using LLM
def get_final_feedback(matches):
    feedback_prompt = f"""
    Based on the following match results:
    Skill Match: {matches['Skill Match']}
    Experience Match: {matches['Experience Match']}
    Education Fit: {matches['Education Fit']}
    
    Provide a detailed evaluation of the candidate's suitability for the job based on these metrics. Include strengths, weaknesses, and any recommendations for improving the resume.
    """
    return get_gemini_response(feedback_prompt)

# Streamlit UI
st.title("DDS Smart ATS")
st.text("Improve your ATS resume score Match")
jd = st.text_area("Paste job description here")
uploaded_file = st.file_uploader("Upload your resume", type="pdf", help="Please upload the PDF")

submit = st.button('Check Your Score')
if submit:
    if uploaded_file is not None:
        # Extract text from the resume
        resume_text = input_pdf_text(uploaded_file)

        # Classify sections of the resume
        resume_sections = classify_sections(resume_text)

        # Classify sections of the job description
        jd_sections = classify_sections(jd)

        # Create Pinecone vector store
        pinecone_store = PineconeVectorStore(index_name=index_name, embedding_function=embedding_function)

        # Create the index using LlamaIndex
        index = VectorStoreIndex(pinecone_store)

        # Upsert embeddings for each section into Pinecone
        for section in resume_sections:
            section_doc = Document(text=resume_sections[section])
            index.add_documents([section_doc], namespace=section)  # Use section name as namespace

        for section in jd_sections:
            section_doc = Document(text=jd_sections[section])
            index.add_documents([section_doc], namespace=section)  # Use section name as namespace

        # Perform similarity searches for each section
        match_results = {}
        for section in resume_sections.keys():
            resume_doc = Document(text=resume_sections[section])
            query_results = index.query(resume_doc, namespace=section)  # Query for the specific section
            matched_content = query_results[0].text if query_results else "No match found."
            match_results[section] = matched_content

        # Calculate matches
        matches = calculate_matches(resume_sections, jd_sections)

        # Get final feedback from LLM
        final_feedback = get_final_feedback(matches)

        # Prepare evaluation feedback
        feedback = []
        for section in match_results:
            feedback.append(f"**{section} Match:** {match_results[section]}")

        # Display results
        st.subheader("Match Results:")
        for result in feedback:
            st.markdown(result)

        # Display match metrics
        st.subheader("Match Metrics:")
        for metric, result in matches.items():
            st.markdown(f"**{metric}:** {result}")

        # Display final feedback
        st.subheader("Final Feedback:")
        st.markdown(final_feedback)
