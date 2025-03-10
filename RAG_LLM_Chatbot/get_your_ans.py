import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from vllm import LLM, SamplingParams
import chromadb
 

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using pdfplumber2."""
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"  # Extract text from each page

    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """
    Splits text into chunks of size `chunk_size` with `overlap` for better context retention.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Max tokens per chunk
        chunk_overlap=overlap,  # Overlapping tokens for continuity
        length_function=len,  # Use character length for splitting
        separators=["\n\n", "\n", " ", ""],  # Tries to split at paragraphs, lines, words
    )

    chunks = text_splitter.split_text(text)
    return chunks

def generate_embeddings(text_chunks):
    """
    Converts text chunks into vector embeddings using a pre-trained model.
    """
    embeddings = embedding_model.encode(text_chunks, convert_to_tensor=True)
    return embeddings

def query_chroma(query, collection, top_k=3):
    """
    Searches ChromaDB for the most relevant chunks based on a query.

    Args:
        query (str): The user question.
        collection: ChromaDB collection containing stored embeddings.
        top_k (int): Number of top results to retrieve.

    Returns:
        List of retrieved documents with scores.
    """
    # Generate an embedding for the query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).tolist()

    # Perform similarity search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k  # Retrieve top_k most relevant chunks
    )

    return results

def your_best_answer(pdf_path, query):
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text,chunk_size=500, overlap=100)
    embeddings = generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    chroma_client = chromadb.PersistentClient(path="/content/rag_test/chroma_db")  # Stores embeddings on disk
    collection = chroma_client.get_or_create_collection(name="pdf_chunks")
    
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],  # Unique ID for each chunk
            documents=[chunk],  # The actual chunk of text
            metadatas=[{"source": "pdf", "chunk_id": i}]  # Optional metadata
        )

    print("✅ Stored embeddings in ChromaDB!")
    
    retrieved_chunks = query_chroma(query, collection,top_k = 3)
    
    return retrieved_chunks
    
    
    