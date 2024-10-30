import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Interview Video Analysis System",
    page_icon="🎥",
)

# Page header
st.markdown("<h1 style='text-align: center;'>Interview Video Analysis System</h1>", unsafe_allow_html=True)

st.write(
    """This system uses AI to analyze interview videos and generate a concise summary of the candidate's responses.
    It highlights key traits demonstrated during the interview, including communication style, active listening, 
    and engagement with the interviewer.
    """
)

# Centered upload container
st.markdown('<div class="upload-container">', unsafe_allow_html=True)

# Video Upload Section
st.markdown("<h4 style='text-align: center;'>Upload Interview Video</h4>", unsafe_allow_html=True)
uploaded_video = st.file_uploader("Upload your interview video in MP4 format.", type=["mp4"], key="video_uploader")

# Process video upload
if uploaded_video is not None:
    st.success("Video uploaded successfully!")
    st.video(uploaded_video)  # Display the video for confirmation
    
    st.write("Processing the interview data... Please wait for a summary.")
    
    # Call the FastAPI endpoint
    api_url = "http://localhost:8003/video"  # Update with your FastAPI endpoint URL
    files = {"file": (uploaded_video.name, uploaded_video, "video/mp4")}
    
    try:
        response = requests.post(api_url, files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()

        # Displaying the interview summary
        st.markdown("### Interview Summary")
        st.markdown(result["summary"])
        st.success("Audio extraction successful!")
        st.write(f"Audio file saved at: {result['audio_file']}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an interview video to see the summary.")

st.markdown('</div>', unsafe_allow_html=True)
