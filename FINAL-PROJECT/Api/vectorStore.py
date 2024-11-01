import logging.config
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec  # type: ignore
import json
import os
from dotenv import load_dotenv  # type: ignore
from uuid import uuid4


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

def initialize_pinecone():
    """Initializes the Pinecone client and ensures the index exists."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Create index if it does not exist
    if 'job-matching' not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name="job-matching",
            dimension=384,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Replace with the correct region
            )
        )
    return pc.Index("job-matching")  # Removed namespace parameter

def store_in_pinecone(document, document_type):
    """Store sections in Pinecone for both job descriptions and resumes."""
    
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    index = initialize_pinecone()  # Ensure the index is initialized
    logging.info("Created Pinecone index for document matching.")
    embeddings_data = []
    logging.info("resume sections: " + str(document))
    cleaned_string = document[7:-3]
    logging.info("cleaned string for adding to db")
    # Check if the document is a JD or a resume
    if document_type.lower() == "job description":
        # Assuming document is in JSON format similar to your example
        sections = json.loads(cleaned_string)  # Load JSON from the document

        # Loop through each JD section
        for jd_name, jd_sections in sections.items():
            logging.info(f"Processing Job Description: {jd_name}")
            for section_key, section_value in jd_sections.items():
                # logging.info(f"Processing section: {section_key}")
                # logging.info(f"Section content: {section_value}")
                
                # Create a unique ID for each section
                section_id = f"jd-{jd_name.lower().replace(' ', '-')}-{section_key.lower().replace(' ', '-')}-{str(uuid4())}"
                
                # Convert the section to JSON string for storage
                section_content = json.dumps({section_key: section_value})

                # Generate embeddings for the section content
                embedding = model.encode(section_content).tolist()

                # Prepare data for upsert
                embeddings_data.append((section_id, embedding, {
                    "content": section_content,
                    "document_type": document_type,
                    "section_name": section_key,
                    "doc_name": jd_name
                }))

    elif document_type.lower() == "resume":
        # Assuming the resume is structured similarly to the JD
        sections = json.loads(cleaned_string)  # Load JSON from the document

        # Loop through each section of the resume
        logging.info(f"Processing Resume")
        for section_key, section_value in sections.items():
            logging.info(f"Processing section: {section_key}")
            logging.info(f"Section content: {section_value}")

            # Create a unique ID for each section
            section_id = f"resume-{section_key.lower().replace(' ', '-')}-{str(uuid4())}"
            
            # Convert the section to JSON string for storage
            section_content = json.dumps({section_key: section_value})

            # Generate embeddings for the section content
            embedding = model.encode(section_content).tolist()

            # Prepare data for upsert
            embeddings_data.append((section_id, embedding, {
                "content": section_content,
                "document_type": document_type,
                "section_name": section_key
            }))

    # Upsert the vectors to the Pinecone index
    index.upsert(vectors=embeddings_data)
    logging.info(f"Stored {len(embeddings_data)} sections in Pinecone for {document_type}.")