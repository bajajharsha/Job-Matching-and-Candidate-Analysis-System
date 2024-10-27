from fastapi import FastAPI
import uvicorn
import logging
import json
import logging.config

# Load logging configuration from the JSON file
with open("FINAL/API/logging_config.json", "r") as file:
    config = json.load(file)
    logging.config.dictConfig(config)

# Initialize FastAPI app
app = FastAPI(
    title="Simple Logging API",
    version="1.0",
    description="A simple FastAPI server with logging."
)

@app.get("/")
async def root():
    logging.info("Root endpoint called.")  # Use logging directly
    return {"message": "Welcome to the Simple Logging API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
