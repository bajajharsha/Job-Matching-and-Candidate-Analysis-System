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
    api_url = "http://0.0.0.0:8000/upload/"  # Replace with your actual API endpoint
    files = {}

    # Add resume to the files dictionary
    if resume_file is not None:
        # send bytes of the file
        files['resume'] = (resume_file.name, resume_file.getvalue(), 'application/pdf')
        
    with open(resume_file.name, 'rb') as resume_file:
        files = {
            'resume': (resume_file.name, resume_file.read()),  # Send resume as bytes
        }

    # Add job descriptions to the files dictionary
    # if jd_files is not None:
    #     for i, jd_file in enumerate(jd_files):
    #         files[f'job_description_{i}'] = (jd_file.name, jd_file.getvalue(), 'text/plain')
    
    print(files)
    # Make the API request
    response = requests.post(api_url, files=files)
    
    return response

if st.button("MATCH", key="match_button"):
    if uploaded_resume and uploaded_jd:
        # print("resume uploaded", uploaded_resume)
        response = api_call(uploaded_resume, uploaded_jd)
        # print(response)
        # response = requests.post(api_url, files=files)

        if response.status_code == 200:
            # Handle successful response
            st.success("Matching process completed successfully!")
            st.json(response.json())  # Display response from the API
        # else:
        #     st.error(f"Error during matching process: {response.text}")

        st.success("Matching process initiated! Please wait...")
        # Add your API call logic here
    else:
        st.warning("Please upload both a resume and a job description before matching.")

def api_call():
    pass