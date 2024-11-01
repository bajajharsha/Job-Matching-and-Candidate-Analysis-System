from vectorStore import initialize_pinecone
import logging.config
from sentence_transformers import SentenceTransformer
import json
from uuid import uuid4
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

def find_best_job_match(resume_sections):
    """Find the best job match for a given resume."""
    index = initialize_pinecone()  # Ensure the index is initialized
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    cleaned_string = resume_sections[9:-3]
    logging.info("resume sections: " + str(cleaned_string))
    sections = eval(cleaned_string)  # Load JSON from the document
    
    final_result = {}
    
    # Loop through each section of the resume
    for section_key, section_value in sections.items():
        logging.info(f"Processing section: {section_key}")
        logging.info(f"Section content: {section_value}")

        # Create a unique ID for each section
        section_id = f"resume-{section_key.lower().replace(' ', '-')}-{str(uuid4())}"
        
        # Convert the section to JSON string for storage
        section_content = json.dumps({section_key: section_value})

        # Generate embeddings for the section content
        query_embedding = model.encode(section_content).tolist()
        result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        # Store results in final_result with section ID as the key
        final_result[section_id] = {
            "section": section_key,
            "matches": result['matches']  # Store the top matching results for the section
        }
        
        # logging.info("final result: " + str(final_result))        
        
    return final_result