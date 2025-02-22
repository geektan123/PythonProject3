import chromadb
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Optional
import tempfile
import mimetypes

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file! Please add GOOGLE_API_KEY to your .env file.")

# Initialize Chroma client
chroma_client = chromadb.Client()
collection_name = "documents"  # Renamed to be more generic

# Initialize Embeddings Model (shared across endpoints)
embedding = GoogleGenerativeAIEmbeddings(
    model='models/text-embedding-004',
    google_api_key=api_key
)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".json", ".txt"}

# Function to load documents based on file type
def load_document(file_path: str, file_extension: str):
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path)
    elif file_extension == ".json":
        loader = JSONLoader(file_path, jq_schema=".content")  # Adjust jq_schema as needed
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return loader.load()

# Endpoint 1: Upload and process document
@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported!")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(await file.read())
        file_path = temp_file.name

    try:
        # Load and split document
        docs = load_document(file_path, file_extension)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.split_documents(docs)

        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document!")

        # Generate embeddings
        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = embedding.embed_documents(doc_texts)

        # Create or reset Chroma collection
        try:
            chroma_client.delete_collection(collection_name)  # Reset for new document
        except:
            pass  # Ignore if collection doesn't exist
        collection = chroma_client.create_collection(collection_name)

        # Store in Chroma
        collection.add(
            embeddings=doc_embeddings,
            documents=doc_texts,
            ids=[f"doc_{i}" for i in range(len(doc_texts))]
        )

        return {"message": "Document processed and embeddings stored successfully", "chunk_count": len(doc_texts)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# Endpoint 2: Query the stored document (unchanged except for name)
@app.post("/query-document/")
async def query_document(query: str = Form(...)):
    try:
        # Get Chroma collection
        collection = chroma_client.get_collection(collection_name)
    except:
        raise HTTPException(status_code=404, detail="No document has been uploaded yet. Please upload a document first.")

    try:
        # Embed the query
        query_embedding = embedding.embed_query(query)

        # Query Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        # Extract results
        most_similar_text = results['documents'][0][0]
        similarity_score = 1 - results['distances'][0][0]  # Convert distance to similarity

        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key
        )

        # Craft prompt
        prompt = (
            f"Context: I have analyzed a stored document. "
            f"The most relevant chunk to the query '{query}' is: '{most_similar_text}' "
            f"(similarity score: {similarity_score:.4f}). Based on this, provide a clear and concise answer to the query."
        )

        # Get response
        response = llm.invoke(prompt)
        gemini_answer = response.content

        # Prepare response
        result = {
            "query": query,
            "most_similar_chunk": most_similar_text,
            "similarity_score": float(similarity_score),
            "answer": gemini_answer
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)