from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from moviepy.editor import VideoFileClip
import assemblyai as aai
import google.generativeai as genai
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Video Analysis API",
    version="1.0",
    description="API for analyzing interview videos."
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("/home/harsha/Desktop/project")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Set AssemblyAI API key
aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY")

def analyze_video(mp3_file):
    """Transcribe audio file using AssemblyAI."""
    logging.info("Starting AssemblyAI transcription...")
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)

    try:
        transcript = transcriber.transcribe(mp3_file, config)
        
        if transcript.status == aai.TranscriptStatus.error:
            logging.error("Transcription failed: %s", transcript.error)
            return {"error": transcript.error}
        
        logging.info("Transcription successful. Transcript: %s", transcript.text)
        return {"transcript": transcript.text}
    
    except Exception as e:
        logging.exception("Error during transcription")
        return {"error": str(e)}

def generate_summary(transcript):
    """Generate a summary of the interview transcript using Google Gemini."""
    genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

    prompt = f"""
    Task: Generate a concise summary of the candidate's interview responses based on the provided transcript. 
    Focus on highlighting the following key traits demonstrated by the candidate:

    1. Communication Style: Describe how the candidate expresses their ideas and conveys information.
    2. Active Listening: Evaluate the candidate’s ability to understand and respond appropriately to questions.
    3. Engagement with the Interviewer: Assess the level of interaction and rapport established during the interview.

    Instructions: 
    - Provide a summary that is clear and concise, ideally within 150-200 words.
    - Use bullet points or short paragraphs for each key trait to ensure easy readability.
    - Focus on specific examples from the transcript that illustrate each trait effectively.

    Transcript: 
    {transcript}
    """
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    logging.info("Generated summary: %s", response.text)
    return response.text    

@app.post("/video")
async def upload_video(file: UploadFile = File(...)):
    """Endpoint to upload video and process transcription."""
    try:
        # Define file paths
        mp4_file = UPLOAD_DIR / file.filename
        mp3_file = mp4_file.with_suffix('.mp3')

        # Save uploaded video
        with open(mp4_file, "wb") as buffer:
            buffer.write(await file.read())

        # Extract audio from video
        video_clip = VideoFileClip(str(mp4_file))
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(str(mp3_file))

        # Close video and audio clips
        audio_clip.close()
        video_clip.close()
        
        # Perform transcription analysis
        transcription_result = analyze_video(str(mp3_file))
        
        if "error" in transcription_result:
            raise HTTPException(status_code=500, detail=transcription_result["error"])
        
        summary = generate_summary(transcription_result["transcript"])
        
        # Delete the extracted audio file and the uploaded video file
        os.remove(mp3_file)
        os.remove(mp4_file)

        # Return the transcript and summary in the response
        return JSONResponse(content={
            "detail": "Audio extraction and transcription successful!",
            "transcript": transcription_result["transcript"],
            "summary": summary,
            "audio_file": str(mp3_file)
        })

    except Exception as e:
        logging.exception("Error processing video")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
