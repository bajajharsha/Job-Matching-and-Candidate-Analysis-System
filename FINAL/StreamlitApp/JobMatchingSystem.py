import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Job Matching and Candidate Analysis System",
    page_icon="👋",
)

st.markdown("<h1 style='text-align: center;'>Job Matching and Candidate Analysis System</h1>", unsafe_allow_html=True)


st.write(
    """This is a job matching system that uses AI to analyze candidate profiles and job descriptions. 
    The system will automatically match candidates to job openings based on key data points like skills, 
    work experience, and qualifications.
    """
)

st.markdown(
    """
    <style>
    .upload-container {
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
    }
    .file-uploader {
        width: 300px;  /* Adjust the width as needed */
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Centered upload container
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

st.markdown("<h4>Upload Resume</h4>", unsafe_allow_html=True)
uploaded_resume = st.file_uploader('', type=['docx', 'pdf'], label_visibility="collapsed", key="resume_uploader", help="Upload your resume in PDF or DOCX format.")

st.markdown("<h4>Upload Job Description</h4>", unsafe_allow_html=True)
uploaded_jd = st.file_uploader('', type=['txt', 'json'], label_visibility="collapsed", key="jd_uploader", help="Upload the job description in TXT or JSON format.", accept_multiple_files=True)

st.markdown('</div>', unsafe_allow_html=True)

# Function to handle API call
def api_call(resume_file, jd_files):
    api_url = "http://0.0.0.0:8002/upload/"  # Replace with your actual API endpoint
    files = {}

    # Process resume
    if resume_file is not None:     
        if resume_file.type == 'application/pdf':
            files['resume'] = (resume_file.name, resume_file.getvalue(), 'application/pdf')
        elif resume_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            files['resume'] = (resume_file.name, resume_file.getvalue(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
        else:
            raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

    # Process job description
    if jd_files is not None:
        for jd_file in jd_files:
            if jd_file.type == 'text/plain':
                files['jds'] = (jd_file.name, jd_file.getvalue(), 'text/plain')
            elif jd_file.type == 'application/json':
                files['jds'] = (jd_file.name, jd_file.getvalue(), 'application/json')
            else:
                raise ValueError("Unsupported file type. Please upload a TXT or JSON file.")
    
    # Make the API request
    response = requests.post(api_url, files=files)
    
    return response

if st.button("MATCH", key="match_button"):
    if uploaded_resume and uploaded_jd:
        st.success("Matching process initiated! Please wait...")
        response = api_call(uploaded_resume, uploaded_jd)
        if response.status_code == 200:
            st.success("Matching process completed successfully!")
            st.json(response)
    else:
        st.warning("Please upload both a resume and a job description before matching.")
