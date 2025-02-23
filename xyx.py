import chromadb
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import Optional, Dict, List
import tempfile
import json
import statistics
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file! Please add GOOGLE_API_KEY to your .env file.")

# Initialize Chroma client
chroma_client = chromadb.Client()
collection_name = "documents"

# Initialize Embeddings Model (shared across endpoints)
embedding = GoogleGenerativeAIEmbeddings(
    model='models/text-embedding-004',
    google_api_key=api_key
)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".json", ".txt"}


# Function to load documents based on file type
def load_document(file_path: str, file_extension: str):
    try:
        if file_extension == ".pdf":
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                raise ValueError("PDF is encrypted and cannot be processed without a password.")
            if len(reader.pages) == 0:
                raise ValueError("PDF has no pages.")

            # Try PyPDFLoader first
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if docs and any(doc.page_content.strip() for doc in docs):
                return docs

            # If no text is extracted, fallback to OCR
            try:
                images = convert_from_path(file_path)
                text = ""
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
                if not text.strip():
                    raise ValueError("No text extracted via OCR. PDF may be empty or unreadable.")
                return [Document(page_content=text, metadata={"source": file_path, "type": "pdf", "ocr_applied": True})]
            except Exception as ocr_error:
                raise ValueError(f"OCR processing failed: {str(ocr_error)}")

        elif file_extension == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension == ".json":
            with open(file_path, "r") as f:
                raw_json = json.load(f)
            return [Document(page_content=json.dumps(raw_json), metadata={"type": "json"})]
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        return loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load document: {str(e)}")


# Function to split documents (skip JSON)
def split_documents(docs: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = []
    for doc in docs:
        if doc.metadata.get("type") == "json":
            split_docs.append(doc)
        else:
            split_docs.extend(text_splitter.split_documents([doc]))
    return split_docs


# Function to extract values from JSON data for aggregation (handles all types)
def extract_values(json_data: List[Dict], field: str) -> tuple:
    values = []
    field_type = None

    for item in json_data:
        try:
            value = item[field]
            if value is None:
                continue

            if field_type is None:
                if isinstance(value, (int, float)):
                    field_type = "numerical"
                elif isinstance(value, str):
                    try:
                        datetime.strptime(value, "%Y-%m-%d")
                        field_type = "date"
                    except ValueError:
                        field_type = "string"
                else:
                    field_type = "string"

            if field_type == "numerical":
                values.append(float(value))
            elif field_type == "date":
                values.append(datetime.strptime(value, "%Y-%m-%d"))
            elif field_type == "string":
                values.append(str(value))
        except KeyError:
            continue

    if not values:
        raise ValueError(f"No valid values found for field '{field}' in the JSON data.")

    return values, field_type


# Endpoint 1: Upload and process document
@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Only {', '.join(SUPPORTED_EXTENSIONS)} files are supported!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(await file.read())
        file_path = temp_file.name

    try:
        docs = load_document(file_path, file_extension)
        docs = split_documents(docs)

        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document!")

        doc_texts = [doc.page_content for doc in docs]
        doc_embeddings = embedding.embed_documents(doc_texts)

        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        collection = chroma_client.create_collection(collection_name)

        collection.add(
            embeddings=doc_embeddings,
            documents=doc_texts,
            metadatas=[{"type": doc.metadata.get("type", "text"), "ocr_applied": doc.metadata.get("ocr_applied", False)}
                       for doc in docs],
            ids=[f"doc_{i}" for i in range(len(doc_texts))]
        )

        return {"message": "Document processed and embeddings stored successfully", "chunk_count": len(doc_texts)}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# Endpoint 2: Query non-JSON documents (PDF, DOCX, TXT)
@app.post("/query-document/")
async def query_document(query: str = Form(...)):
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        raise HTTPException(status_code=404,
                            detail="No document has been uploaded yet. Please upload a document first.")

    # Check if the document is JSON
    results = collection.get(include=["metadatas"])
    is_json = any(meta.get("type") == "json" for meta in results["metadatas"])
    if is_json:
        raise HTTPException(status_code=400,
                            detail="This endpoint is for non-JSON documents only. Use /query-json/ for JSON documents.")

    try:
        query_embedding = embedding.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        most_similar_text = results['documents'][0][0]
        similarity_score = 1 - results['distances'][0][0]

        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key
        )
        prompt = (
            f"Context: I have analyzed a stored document. "
            f"The most relevant chunk to the query '{query}' is: '{most_similar_text}' "
            f"(similarity score: {similarity_score:.4f}). Based on this, provide a clear and concise answer to the query."
        )
        response = llm.invoke(prompt)
        gemini_answer = response.content

        return {
            "query": query,
            "most_similar_chunk": most_similar_text,
            "similarity_score": float(similarity_score),
            "answer": gemini_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Endpoint 3: Query JSON documents (supports all field types)
@app.post("/query-json/")
async def query_json(
        operation: str = Form(..., description="Operation: max, min, sum, avg"),
        field: str = Form(..., description="Field name to aggregate")
):
    try:
        collection = chroma_client.get_collection(collection_name)
        results = collection.get(include=["documents", "metadatas"])
        json_docs = [doc for doc, meta in zip(results["documents"], results["metadatas"]) if meta.get("type") == "json"]

        if not json_docs:
            raise HTTPException(status_code=404,
                                detail="No JSON data has been uploaded yet. Please upload a JSON file first.")

        json_data = []
        for doc in json_docs:
            json_data.extend(json.loads(doc))

        values, field_type = extract_values(json_data, field)

        if operation in ("max", "min"):
            result = max(values) if operation == "max" else min(values)
            if field_type == "date":
                result = result.strftime("%Y-%m-%d")
            elif field_type == "numerical":
                result = float(result)
            else:
                result = str(result)
        elif field_type == "numerical" and operation == "sum":
            result = float(sum(values))
        elif field_type == "numerical" and operation == "avg":
            result = float(statistics.mean(values))
        else:
            raise HTTPException(status_code=400,
                                detail=f"Operation '{operation}' is not supported for {field_type} field '{field}'. "
                                       "Use 'max' or 'min' for strings/dates, and 'max', 'min', 'sum', or 'avg' for numbers.")

        return {
            "operation": operation,
            "field": field,
            "field_type": field_type,
            "result": result,
            "value_count": len(values)
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing JSON query: {str(e)}")


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)