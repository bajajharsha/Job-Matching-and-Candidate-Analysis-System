from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec
import openai

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone client
pc = Pinecone(
    api_key="c5e1cdaf-7368-4800-9239-9ebfe1dfe34c",  # Use your actual API key
    environment="us-east-1"  # Replace with your actual environment
)

# Check if the index already exists and create if necessary
index_name = "project"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Match this to the embedding model's output dimension
        metric='cosine',  # You can change the metric based on your needs
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Replace with the correct region
        )
    )

# Access the existing index
index = pc.Index(index_name)

# Example data: resumes and job descriptions
resumes = [
    "John Doe\nSoftware Engineer with 3 years of experience in full-stack development. Proficient in Python, Java, and JavaScript, with a strong understanding of web frameworks such as Django and React. Skilled in developing REST APIs and deploying applications on AWS."]

job_descriptions = [
    "Software Engineer - Full Stack\nWe are seeking a Full Stack Software Engineer with a strong background in both frontend and backend development. Required skills include Python, JavaScript, and experience with Django or similar frameworks. Familiarity with cloud services like AWS is a plus. You will work on building scalable web applications in a collaborative team environment.",
    "Data Analyst\nSeeking a Data Analyst with expertise in data visualization and analytics. Required skills include SQL, Python, and experience with data visualization tools like Tableau or Power BI. Candidates should be able to analyze large datasets and create actionable insights for business growth. Experience in machine learning is preferred."
]

# Prepare upsert data for resumes
upserted_data = []
for i, resume in enumerate(resumes):
    embedding = model.encode(resume).tolist()
    upserted_data.append((f"resume-{i}", embedding, {"content": resume}))

# Prepare upsert data for job descriptions
for i, job in enumerate(job_descriptions):
    embedding = model.encode(job).tolist()
    upserted_data.append((f"job-{i}", embedding, {"content": job}))

# Upsert all data at once
index.upsert(vectors=upserted_data)

# Now we will query using a specific resume to find the best job match
query_resume = resumes[0]  # Using the first resume as the query
query_embedding = model.encode(query_resume).tolist()

# Perform the query
result = index.query(vector=query_embedding, top_k=2, include_metadata=True)

# Print the result
print("Query Result:", result)

# Define the system role for GPT-4
system_role = (
    "Answer the question as truthfully as possible using the provided context, "
    "and if the answer is not contained within the text and requires some latest information to be updated, "
    "print 'Sorry Not Sufficient context to answer query' \n"
)

# Check if there are matches in the result
if result['matches']:
    context = result['matches'][0]['metadata']['content']
    
    # Prepare the user input for GPT-4
    user_input = context + '\n' + "What job matches this resume?"  # Adjust the question as needed
    
    # Call GPT-4
    gpt4_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_input}
        ]
    )
    
    # Print the GPT-4 response
    print("GPT-4 Response:", gpt4_response['choices'][0]['message']['content'])
else:
    print("Sorry Not Sufficient context to answer query")