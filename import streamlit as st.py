import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import pinecone
import numpy as np  # Make sure to install NumPy

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-west1-gcp")  # Change the environment as needed
index_name = "job-resume-matching"  # Your Pinecone index name
index = pinecone.Index(index_name)

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

# Function to create embeddings (assume you have a function to generate embeddings)
def create_embedding(text):
    # Here you would replace this with your actual embedding creation logic
    # For instance, using Hugging Face models or any other method
    return np.random.rand(512).tolist()  # Placeholder: Replace with actual embeddings

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

        # Create embeddings for the resume and job description
        resume_embedding = create_embedding(resume_text)
        jd_embedding = create_embedding(jd)

        # Store embeddings in Pinecone
        index.upsert([(uploaded_file.name, resume_embedding)])  # Upsert resume embedding
        index.upsert([("jd_description", jd_embedding)])  # Upsert job description embedding

        # Perform similarity search
        query_results = index.query(queries=[resume_embedding], top_k=1)  # Find the most similar JD
        matched_jd_id = query_results["matches"][0]["id"]  # Get the ID of the matched JD
        
        # Fetch the matched JD content (assuming you have stored it somewhere)
        matched_jd_content = jd  # In a real case, you would retrieve this from your storage

        # Prepare the input for the LLM
        input_prompt = f"""
        ### Your existing prompt goes here.
        resume={resume_text}
        jd={matched_jd_content}
        ### Evaluation Output:
        1. Calculate the percentage of match between the resume and the job description. Give a number and some explanation
        2. Identify any key keywords that are missing from the resume in comparison to the job description.
        3. Offer specific and actionable tips to enhance the resume and improve its alignment with the job requirements.
        """
        
        # Get evaluation response from Gemini
        response = get_gemini_response(input_prompt)
        st.subheader(response)

# Make sure to clean up the index or manage the embeddings as necessary
