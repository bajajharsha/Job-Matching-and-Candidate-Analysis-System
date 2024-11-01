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
    api_url = "http://0.0.0.0:8001/upload/"  # Replace with your actual API endpoint
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
    
    # st.write("Payload::: " , files)
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
                response_json = response.json()

                # Displaying structured response
                st.header("Matching Results")
                st.markdown("---")  # Horizontal line for separation

                # Best Job Match Section
                best_fit = response_json.get("BestJobMatch", {})
                st.subheader("Best Job Match")
                if best_fit:
                    st.markdown(f"**Job Description:** {best_fit.get('jobDescription', 'N/A')}")
                    st.markdown(f"**Reason:** {best_fit.get('reason', 'N/A')}")

                    # Analytics for Best Job Match
                    analytics = best_fit.get("analytics", {})
                    if analytics:
                        st.subheader("Analytics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Skills Analysis:** " + analytics.get('skillsAnalysis', 'N/A'))
                            st.write("**Experience Analysis:** " + analytics.get('experienceAnalysis', 'N/A'))
                        with col2:
                            st.write("**Education Analysis:** " + analytics.get('educationAnalysis', 'N/A'))
                            st.write("**Technology Analysis:** " + analytics.get('technologyAnalysis', 'N/A'))
                else:
                    st.write("None")

                st.markdown("---")  # Horizontal line for separation

                # Skill Match Section
                skill_match = response_json.get("Skill Match", {})
                st.subheader("Skill Match")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Match Percentage:** {skill_match.get('matchPercentage', 'N/A')}")
                    st.markdown(f"**Description:** {skill_match.get('description', 'N/A')}")
                with col2:
                    matched_skills = skill_match.get("matchedSkills", {}).get("skills", [])
                    missing_skills = skill_match.get("missingSkills", {}).get("skills", [])
                    st.write("**Matched Skills:** " + (", ".join(matched_skills) if matched_skills else "None"))
                    st.write("**Missing Skills:** " + (", ".join(missing_skills) if missing_skills else "None"))

                st.markdown("---")  # Horizontal line for separation

                # Experience Match Section
                experience_match = response_json.get("Experience Match", {})
                st.subheader("Experience Match")
                overall_relevance = experience_match.get("overallRelevance", {})
                st.markdown(f"**Overall Relevance:** {overall_relevance.get('percentage', 'N/A')}")

                relevant_jobs = experience_match.get("relevantExperience", {}).get("jobs", [])
                if relevant_jobs:
                    st.write("**Relevant Past Experience:**")
                    for job in relevant_jobs:
                        st.write(f"- **Job Title:** {job.get('jobTitle', 'N/A')}")
                        st.write(f"  **Relevance:** {job.get('relevance', 'N/A')}")
                        st.write(f"  **Details:** {job.get('details', 'N/A')}")
                else:
                    st.write("None")

                responsibility_overlap = experience_match.get("responsibilityOverlap", {}).get("responsibilities", [])
                st.write("**Matched Responsibilities:** " + (", ".join(responsibility_overlap) if responsibility_overlap else "None"))

                st.markdown("---")  # Horizontal line for separation

                # Education Fit Section
                education_fit = response_json.get("Education Fit", {})
                st.subheader("Education Fit")
                st.markdown(f"**Match Percentage:** {education_fit.get('matchPercentage', 'N/A')}")
                st.markdown(f"**Description:** {education_fit.get('description', 'N/A')}")

                matched_qualifications = education_fit.get("matchedQualifications", {}).get("qualifications", [])
                missing_qualifications = education_fit.get("missingQualifications", {}).get("qualifications", [])
                st.write("**Matched Qualifications:** " + (", ".join(matched_qualifications) if matched_qualifications else "None"))
                st.write("**Missing Qualifications:** " + (", ".join(missing_qualifications) if missing_qualifications else "None"))

                st.markdown("---")  # Horizontal line for separation

                # Technological Fit Section
                tech_fit = response_json.get("Technological Fit", {})
                st.subheader("Technological Fit")
                st.markdown(f"**Match Percentage:** {tech_fit.get('matchPercentage', 'N/A')}")
                st.markdown(f"**Description:** {tech_fit.get('description', 'N/A')}")

                matched_technologies = tech_fit.get("matchedTechnologies", {}).get("technologies", [])
                missing_technologies = tech_fit.get("missingTechnologies", {}).get("technologies", [])
                st.write("**Matched Technologies:** " + (", ".join(matched_technologies) if matched_technologies else "None"))
                st.write("**Missing Technologies:** " + (", ".join(missing_technologies) if missing_technologies else "None"))

            else:
                st.error("Failed to complete the matching process. Please try again.")
    else:
        st.warning("Please upload both a resume and a job description before matching.")