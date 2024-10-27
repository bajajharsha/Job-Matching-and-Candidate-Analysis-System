import os
import pickle
import tempfile
from io import BytesIO
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

class ChatWithDoc:
    """
    Handles conversational interactions with a document base by maintaining a FAISS index for retrieval and processing user queries.
    Parameters:
        - api_key (str): The API key to interact with the OpenAI services.
        - user_id (str): Identifier for the user to create personalized indexes and memory.
    Processing Logic:
        - On initialization, load existing conversation memory or create a new one.
        - Provides methods to update the FAISS index with new documents and load an existing index.
        - Supports conversation retrieval from indexed documents and saves conversations to user memory.
        - Handles different document types with specific loaders and processes them accordingly.
    """
    def __init__(self, api_key: str, user_id: str):
        self.api_key = api_key
        self.user_id = user_id
        self.memory = self.load_memory()
        self.qa_chain = None  # Store QA chain after creation

    def save_memory(self):
        memory_path = f"{self.user_id}_memory.pkl"
        with open(memory_path, "wb") as f:
            pickle.dump(self.memory, f)

    def load_memory(self):
        """Loads and returns the user's conversation memory from a file, if available; otherwise, initializes a new conversation memory.
        Parameters:
            - None
        Returns:
            - ConversationBufferMemory: An instance of the conversation memory, either loaded from a file or newly created.
        Processing Logic:
            - Checks if the user's memory file exists based on their user_id.
            - If the file exists, the user's memory is loaded using pickle.
            - If the file does not exist, a new ConversationBufferMemory object is instantiated with default parameters."""
        memory_path = f"{self.user_id}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, "rb") as f:
                memory = pickle.load(f)
        else:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return memory

    def load_documents(self, file_path, file_extension):
        """Load documents from a file with the specified file extension.
        Parameters:
            - file_path (str): The path to the input file.
            - file_extension (str): The extension of the file used to determine the loading procedure.
        Returns:
            - list: A list of Document objects containing the loaded data.
        Processing Logic:
            - .pdf files are loaded using a PyPDFLoader.
            - .docx files are not supported and raise a ValueError.
            - .xlsx files are converted to strings and saved as Document objects, one per sheet.
            - .csv files are converted to a single string and saved as one Document object."""
        ext = file_extension.lower()
        documents = []

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif ext == ".docx":
            # Assuming you have a custom way to process .docx without unstructured
            raise ValueError("Support for .docx not implemented without unstructured.")
        elif ext == ".xlsx":
            xlsx_file = pd.ExcelFile(file_path)
            for sheet in xlsx_file.sheet_names:
                df = pd.read_excel(xlsx_file, sheet_name=sheet)
                text = df.to_string()
                documents.append(Document(page_content=text))
        elif ext == ".csv":
            csv_data = pd.read_csv(file_path)
            text = csv_data.to_string()
            documents.append(Document(page_content=text))
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return documents

    def update_faiss_index(self, file_path, file_extension):
        # Set the base folder for storing FAISS indexes
        """Updates the FAISS index with new documents and creates a conversational retrieval chain.
        Parameters:
            - file_path (str): Path to the file containing documents to be indexed.
            - file_extension (str): The file extension, used for parsing the document format.
        Returns:
            - ConversationalRetrievalChain: An instance of a conversational retrieval system.
        Processing Logic:
            - The embeddings are generated for the new documents and the FAISS index is updated or created.
            - A text splitter is used to divide the documents into manageable chunks for processing.
            - All folders necessary for storing the FAISS index are ensured to exist.
            - The QA chain is formed using the updated FAISS index and a ConversationalRetrievalChain."""
        base_folder = "faiss"
        user_folder = os.path.join(base_folder, self.user_id)

        embeddings = OpenAIEmbeddings(api_key=self.api_key)

        # Check if the user-specific folder exists and load the vectorstore
        if os.path.exists(user_folder):
            vectorstore = FAISS.load_local(
                user_folder,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vectorstore = None

        # Load documents and split them into chunks
        docs = self.load_documents(file_path, file_extension)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Update the vectorstore
        if vectorstore is None:
            vectorstore = FAISS.from_documents(splits, embeddings)
        else:
            vectorstore.add_documents(splits)

        # Ensure the base folder and user folder exist
        os.makedirs(user_folder, exist_ok=True)

        # Save the updated FAISS index
        vectorstore.save_local(user_folder)

        # Create the QA chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=self.api_key),
            retriever=retriever,
            memory=self.memory
        )

        return self.qa_chain

    def load_existing_faiss_index(self):
        """
        Load the FAISS index if it exists, and return the QA chain.
        """
        base_folder = "faiss"
        user_folder = os.path.join(base_folder, self.user_id)
        embeddings = OpenAIEmbeddings(api_key=self.api_key)

        if os.path.exists(user_folder):
            vectorstore = FAISS.load_local(
                user_folder,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=self.api_key),
                retriever=retriever,
                memory=self.memory
            )
        else:
            raise ValueError("FAISS index does not exist. Load documents first.")

        return self.qa_chain


def loaddoc(file_bytes: bytes, file_extension: str, api_key: str, user_id: str) -> ConversationalRetrievalChain:
    """
    Load documents and update the FAISS index.
   
    Args:
        file_bytes (bytes): The bytes of the document file.
        file_extension (str): The extension of the document file.
        api_key (str): API key for the OpenAI model.
        user_id (str): Unique user identifier.
       
    Returns:
        ConversationalRetrievalChain: The QA chain for the loaded documents.
    """
    # Create a temporary file to store the document content
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name  # Get the path to the temporary file

    chat_doc = ChatWithDoc(api_key, user_id)
    qa_chain = chat_doc.update_faiss_index(temp_file_path, file_extension)

    return qa_chain


def chatwithdoc(query: str, api_key: str, user_id: str) -> str:
    """
    Query the loaded documents using the QA chain.
   
    Args:
        query (str): The question to ask.
        api_key (str): API key for the OpenAI model.
        user_id (str): Unique user identifier.
       
    Returns:
        str: The answer to the query.
    """
    chat_doc = ChatWithDoc(api_key, user_id)

    # Try to load the existing FAISS index and QA chain
    try:
        if chat_doc.qa_chain is None:
            chat_doc.load_existing_faiss_index()
        result = chat_doc.qa_chain({"question": query})
    except ValueError as e:
        return str(e)

    chat_doc.save_memory()
    return result['answer']