import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import List
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import asyncio
import tempfile

# Initialize FastAPI app
app = FastAPI(title="PDF Query AI Agent", description="AI agent for querying uploaded PDF documents without a vector database")

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found in .env file! Please add GOOGLE_API_KEY to your .env file.")

# Initialize Embeddings Model
embedding = GoogleGenerativeAIEmbeddings(
    model='models/text-embedding-004',
    google_api_key=api_key
)

# Global variables for in-memory storage
documents = []
embeddings = []
metadatas = []
pdf_loaded = False
current_pdf_path = None

def load_pdf_document(file_path: str) -> List[Document]:
    """Load PDF document with OCR fallback"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF document not found at: {file_path}")

        reader = PdfReader(file_path)
        if reader.is_encrypted:
            raise ValueError("PDF is encrypted and cannot be processed without a password.")
        if len(reader.pages) == 0:
            raise ValueError("PDF has no pages.")

        # Try PyPDFLoader first
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check if we got meaningful content
        if docs and any(doc.page_content.strip() for doc in docs):
            return docs

        # If no text is extracted, fallback to OCR
        print("No text extracted via standard PDF loader, attempting OCR...")
        try:
            images = convert_from_path(file_path)
            text = ""
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                text += f"Page {i + 1}:\n{page_text}\n\n"

            if not text.strip():
                raise ValueError("No text extracted via OCR. PDF may be empty or unreadable.")

            return [Document(
                page_content=text,
                metadata={"source": file_path, "type": "pdf", "ocr_applied": True}
            )]
        except Exception as ocr_error:
            raise ValueError(f"OCR processing failed: {str(ocr_error)}")

    except Exception as e:
        raise ValueError(f"Failed to load PDF document: {str(e)}")

def split_pdf_documents(docs: List[Document]) -> List[Document]:
    """Split PDF documents into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_documents(docs)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def similarity_search(query_embedding: List[float], k: int = 3) -> List[dict]:
    """Perform in-memory similarity search using cosine similarity"""
    if not embeddings:
        return []

    # Calculate cosine similarity for each embedding
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    
    # Get indices of top-k similarities
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Prepare results
    results = [
        {
            "content": documents[i].page_content,
            "score": similarities[i],
            "metadata": metadatas[i]
        }
        for i in top_k_indices
    ]
    
    return results

async def initialize_pdf(file_path: str):
    """Initialize the PDF document in memory"""
    global pdf_loaded, documents, embeddings, metadatas, current_pdf_path

    if pdf_loaded and current_pdf_path == file_path:
        return

    try:
        # Clear existing data
        documents = []
        embeddings = []
        metadatas = []

        print("Loading PDF document...")
        docs = load_pdf_document(file_path)
        print(f"Loaded {len(docs)} pages from PDF")

        # Split documents
        split_docs = split_pdf_documents(docs)
        print(f"Split into {len(split_docs)} chunks")

        if not split_docs:
            raise ValueError("No content could be extracted from the PDF document!")

        # Generate embeddings
        doc_texts = [doc.page_content for doc in split_docs]
        print("Generating embeddings...")
        doc_embeddings = embedding.embed_documents(doc_texts)

        # Store in memory
        documents = split_docs
        embeddings = doc_embeddings
        metadatas = [{
            "type": "pdf",
            "chunk_id": i,
            "source": file_path,
            "ocr_applied": doc.metadata.get("ocr_applied", False)
        } for i, doc in enumerate(split_docs)]

        pdf_loaded = True
        current_pdf_path = file_path
        print(f"Successfully loaded PDF document with {len(split_docs)} chunks in memory")

    except Exception as e:
        raise RuntimeError(f"Failed to initialize PDF document: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Startup event (no default PDF loading)"""
    pass

@app.get("/")
async def root():
    """Root endpoint with information about the PDF Query AI Agent"""
    return {
        "message": "PDF Query AI Agent",
        "description": "Ask questions about uploaded PDF documents",
        "vector_database": "None (in-memory storage)",
        "endpoints": {
            "/upload-pdf/": "POST - Upload a PDF document",
            "/ask-pdf/": "POST - Ask a question about the uploaded PDF",
            "/health/": "GET - Check system health",
            "/reinitialize-pdf/": "POST - Reinitialize PDF document"
        },
        "pdf_loaded": pdf_loaded
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if pdf_loaded else "no_pdf_loaded",
        "pdf_loaded": pdf_loaded,
        "vector_database": "None (in-memory storage)"
    }

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF document to be processed"""
    global pdf_loaded, current_pdf_path

    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Initialize with the new PDF
        pdf_loaded = False
        await initialize_pdf(temp_file_path)
        
        return {
            "message": "PDF successfully uploaded and processed",
            "filename": file.filename,
            "status": "success",
            "vector_database": "None (in-memory storage)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/ask-pdf/")
async def ask_pdf(question: str = Form(..., description="Your question about the uploaded PDF")):
    """Ask a question about the uploaded PDF document"""
    if not pdf_loaded or not documents:
        raise HTTPException(
            status_code=503,
            detail="No PDF document loaded. Please upload a PDF first."
        )

    try:
        # Generate query embedding
        query_embedding = embedding.embed_query(question)

        # Search for relevant context in memory
        results = await similarity_search(query_embedding, k=3)

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found in the uploaded PDF for your question."
            )

        # Process results
        contexts = [
            {
                "text": result["content"],
                "similarity_score": result["score"],
                "metadata": result["metadata"],
                "rank": i + 1
            }
            for i, result in enumerate(results)
        ]

        # Use the most relevant context for the primary response
        primary_context = contexts[0]

        # Check if we have sufficient confidence
        if primary_context["similarity_score"] < 0.3:
            return {
                "question": question,
                "answer": "I cannot find sufficient relevant information in the uploaded PDF to answer your question confidently. Please rephrase your question or upload a different PDF.",
                "confidence": "low",
                "contexts": contexts[:1],
                "vector_database": "None (in-memory storage)"
            }

        # Generate response using LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.1
        )

        # Create comprehensive prompt
        context_text = "\n\n---\n\n".join([ctx["text"] for ctx in contexts])

        prompt = f"""You are a PDF Query AI agent with access to an uploaded PDF document.

QUESTION: {question}

RELEVANT PDF CONTEXT:
{context_text}

INSTRUCTIONS:
1. Answer the question based ONLY on the provided PDF context
2. Be precise and cite specific sections when possible
3. If the context doesn't fully answer the question, clearly state the limitations
4. Use clear, professional language
5. Do not make assumptions or provide information not found in the context
6. If multiple interpretations are possible, acknowledge this

Provide a comprehensive answer based on the PDF context provided."""

        response = llm.invoke(prompt)
        answer = response.content

        # Determine confidence level
        if primary_context["similarity_score"] >= 0.7:
            confidence = "high"
        elif primary_context["similarity_score"] >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "vector_database": "None (in-memory storage)",
            "primary_context": {
                "text": primary_context["text"][:500] + "..." if len(primary_context["text"]) > 500 else
                primary_context["text"],
                "similarity_score": round(primary_context["similarity_score"], 4)
            },
            "additional_contexts": [
                {
                    "text": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
                    "similarity_score": round(ctx["similarity_score"], 4),
                    "rank": ctx["rank"]
                } for ctx in contexts[1:]
            ],
            "disclaimer": "This response is based on the uploaded PDF and should not be considered as professional advice. Consult a qualified professional for specific situations."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/reinitialize-pdf/")
async def reinitialize_pdf():
    """Reinitialize the current PDF document (useful for updates or troubleshooting)"""
    global pdf_loaded

    try:
        if not current_pdf_path:
            raise HTTPException(status_code=400, detail="No PDF currently loaded")

        pdf_loaded = False
        await initialize_pdf(current_pdf_path)
        return {
            "message": "PDF document successfully reinitialized",
            "status": "success",
            "vector_database": "None (in-memory storage)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reinitialize PDF: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
