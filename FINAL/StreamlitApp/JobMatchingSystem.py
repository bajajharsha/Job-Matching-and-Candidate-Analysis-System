import streamlit as st
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Job Matching and Candidate Analysis System",
    page_icon="👋",
)

# Main header
st.markdown("<h1 style='text-align: center;'>Job Matching and Candidate Analysis System</h1>", unsafe_allow_html=True)

st.write(
    """This job matching system analyzes candidate profiles and job descriptions 
    to determine the best fit based on skills, experience, and qualifications.
    """
)

# Styling for the upload section
st.markdown(
    """
    <style>
    .upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 20px;
    }
    .file-uploader {
        width: 100%;
        max-width: 500px;
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
            st.error("Unsupported resume file type. Please upload a PDF or DOCX file.")
            return None

    # Process job description
    if jd_files is not None:
        for jd_file in jd_files:
            if jd_file.type == 'text/plain':
                files['jds'] = (jd_file.name, jd_file.getvalue(), 'text/plain')
            elif jd_file.type == 'application/json':
                files['jds'] = (jd_file.name, jd_file.getvalue(), 'application/json')
            else:
                st.error("Unsupported job description file type. Please upload a TXT or JSON file.")
                return None
    
    # Make the API request
    try:
        response = requests.post(api_url, files=files)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None

# Match button and progress
# if st.button("MATCH", key="match_button"):
#     if uploaded_resume and uploaded_jd:
#         with st.spinner("Matching process initiated. Please wait..."):
#             response = api_call(uploaded_resume, uploaded_jd)
#             if response and response.status_code == 200:
#                 st.success("Matching process completed successfully!")
#                 # Displaying formatted JSON response
#                 response_json = response.json()
#                 st.json(response_json)
#             else:
#                 st.error("Failed to complete the matching process. Please try again.")
#     else:
#         st.warning("Please upload both a resume and a job description before matching.")


# Match button and progress
if st.button("MATCH", key="match_button"):
    if uploaded_resume and uploaded_jd:
        with st.spinner("Matching process initiated. Please wait..."):
            response = api_call(uploaded_resume, uploaded_jd)
            if response and response.status_code == 200:
                st.success("Matching process completed successfully!")
                
                # Extracting JSON string from the response object
                response_str = response.text  # Use .text to get the response content as a string
                
                # Now parse the JSON string
                res = eval(response_str)
                
                st.json(res)
                response_json = res["detail"]
                st.write(response_json["Skill Match"])
                # Displaying structured response
                st.header("Matching Results")
                
                # Skill Match Section
                skill_match = response_json["Skill Match"] | {}
                skill_match = response_json.get("Skill Match", {})
                st.subheader("Skill Match")
                st.write(f"**Match Percentage**: {skill_match.get('matchPercentage', 'N/A')}")
                st.write(f"**Description**: {skill_match.get('description', 'N/A')}")
                
                matched_skills = skill_match.get("matchedSkills", {}).get("skills", [])
                missing_skills = skill_match.get("missingSkills", {}).get("skills", [])
                
                st.write("**Matched Skills:**")
                st.write(", ".join(matched_skills) if matched_skills else "None")
                st.write("**Missing Skills:**")
                st.write(", ".join(missing_skills) if missing_skills else "None")

                # Experience Match Section
                experience_match = response_json.get("Experience Match", {})
                st.subheader("Experience Match")
                overall_relevance = experience_match.get("overallRelevance", {}).get("percentage", "N/A")
                st.write(f"**Overall Relevance**: {overall_relevance}")
                
                relevant_jobs = experience_match.get("relevantExperience", {}).get("jobs", [])
                if relevant_jobs:
                    st.write("**Relevant Past Experience:**")
                    for job in relevant_jobs:
                        st.write(f"- **Job Title**: {job.get('jobTitle', 'N/A')}")
                        st.write(f"  **Relevance**: {job.get('relevance', 'N/A')}")
                        st.write(f"  **Details**: {job.get('details', 'N/A')}")
                else:
                    st.write("None")

                responsibility_overlap = experience_match.get("responsibilityOverlap", {}).get("responsibilities", [])
                st.write("**Matched Responsibilities:**")
                st.write(", ".join(responsibility_overlap) if responsibility_overlap else "None")

                # Education Fit Section
                education_fit = response_json.get("Education Fit", {})
                st.subheader("Education Fit")
                st.write(f"**Match Percentage**: {education_fit.get('matchPercentage', 'N/A')}")
                matched_qualifications = education_fit.get("matchedQualifications", {}).get("qualifications", [])
                st.write("**Matched Qualifications:**")
                st.write(", ".join(matched_qualifications) if matched_qualifications else "None")

                # Technological Fit Section
                tech_fit = response_json.get("Technological Fit", {})
                st.subheader("Technological Fit")
                st.write(f"**Match Percentage**: {tech_fit.get('matchPercentage', 'N/A')}")
                
                matched_technologies = tech_fit.get("matchedTechnologies", {}).get("technologies", [])
                missing_technologies = tech_fit.get("missingTechnologies", {}).get("technologies", [])
                st.write("**Matched Technologies:**")
                st.write(", ".join(matched_technologies) if matched_technologies else "None")
                st.write("**Missing Technologies:**")
                st.write(", ".join(missing_technologies) if missing_technologies else "None")

                # Additional note
                best_fit = response_json.get("BestJobMatch", "None")  # Default to "None" if not found
                st.write("**Best Fit Job Title:**")
                st.write(best_fit if best_fit else "None")      
            else:
                st.error("Failed to complete the matching process. Please try again.")
    else:
        st.warning("Please upload both a resume and a job description before matching.")
