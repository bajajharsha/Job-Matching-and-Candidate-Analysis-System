import os
from dotenv import load_dotenv # type: ignore
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
huggingface_token = os.getenv("HUGGINGFACE_API_KEY")

from langchain_community.document_loaders import PyPDFLoader # type: ignore
# from langchain_community.document_loaders import Docx2txtLoader # type: ignore

pdf_loader = PyPDFLoader("harsha_bajaj_resumee.pdf")
text_documents_pdf = pdf_loader.load()
print(text_documents_pdf)

resume_text = text_documents_pdf

# docx_loader = Docx2txtLoader("resume_001.docx")
# text_documents_docx = docx_loader.load()
# print(text_documents_docx)

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, model_kwargs={"max_length": 128, "token" : huggingface_token}, temperature = 0.7)

template = """Your task is to analyze the provided resume text and identify key sections. 
These sections may not be explicitly labeled and should be inferred from the content. Use exact content as it is. 
Consider sections such as Summary, Skills, Experience, Education, and Certifications, but also recognize other relevant sections that may emerge from the text.

Here is the extracted resume text:

{resume_text}

Please categorize the information into distinct sections based on your understanding of the content. Provide a clear and organized classification."""

prompt = PromptTemplate(template=template, input_variables=["resume_text"])

llm_chain = prompt | llm

response = llm_chain.invoke({"resume_text": resume_text})
print(response)

# # Load a transformer model for text embedding
# tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")
# model = AutoModel.from_pretrained("all-MiniLM-L6-v2")

# def embed_text(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
#     return embeddings

# # Segregate text into sections dynamically
# def segregate_and_embed(text):
#     # Split the resume text into sentences
#     sentences = text.split('. ')
    
#     # Generate embeddings for each sentence
#     sentence_embeddings = []
#     for sentence in sentences:
#         if len(sentence.strip()) > 0:
#             embedding = embed_text(sentence)
#             sentence_embeddings.append((sentence, embedding))
    
#     # Return the segregated sentences with their embeddings
#     return sentence_embeddings

# # Get segregated sections with embeddings
# segregated_sections = segregate_and_embed(resume_text)

# # Displaying the results
# for section in segregated_sections:
#     print("Section Text:", section[0])
#     print("Embedding:", section[1])
#     print("--------------------------------------------------")

# api_key = None
# CONFIG_PATH = r"config.yaml"

# with open(CONFIG_PATH) as file:
#     data = yaml.load(file, Loader=yaml.FullLoader)
#     api_key = data['OPENAI_API_KEY']

# def ats_extractor(resume_data):

#     prompt = '''
#     You are an AI bot designed to act as a professional for parsing resumes. You are given with resume and your job is to extract the following information from the resume:
#     1. full name
#     2. email id
#     3. github portfolio
#     4. linkedin id
#     5. employment details
#     6. technical skills
#     7. soft skills
#     Give the extracted information in json format only
#     '''

#     openai_client = OpenAI(
#         api_key = api_key
#     )    

#     messages=[
#         {"role": "system",
#         "content": prompt}
#         ]
    
#     user_content = resume_data
    
#     messages.append({"role": "user", "content": user_content})

#     response = openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=messages,
#                 temperature=0.0,
#                 max_tokens=1500)
        
#     data = response.choices[0].message.content

#     #print(data)
#     return data