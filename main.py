import os
import faiss
import PyPDF2
from typing import List
from sentence_transformers import SentenceTransformer
import openai
import numpy as np

# ============ Configuration ============
OPENAI_API_KEY = "*****************"  # REPLACE with your actual key
openai.api_key = OPENAI_API_KEY

PDF_PATH = r"C:\Users\DELL\Desktop\RAG App\sample.pdf"  # Modify your path
CHUNK_SIZE = 300
TOP_K = 3
EMBED_MODEL = "all-MiniLM-L6-v2"

# ============ PDF Text Extraction ============
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"[Error] Failed to read PDF: {e}")
    return text

# ============ Chunking ============
def split_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ============ Embedding ============
def embed_chunks(chunks: List[str], model_name: str):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# ============ FAISS Index ============
def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    return index

# ============ Chunk Retrieval ============
def retrieve_chunks(query: str, chunks: List[str], model, index, top_k: int) -> List[str]:
    query_embedding = model.encode([query])
    query_embedding_np = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding_np, top_k)
    return [chunks[i] for i in indices[0]]

# ============ Generate Response using OpenAI GPT ============
def generate_response(query: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question:

Context:
{context}

Question: {query}
Answer:"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[OpenAI Error] {str(e)}"

# ============ Full Pipeline ============
def rag_pipeline(pdf_path: str, query: str):
    print("[1] Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No text extracted from PDF.")

    print("[2] Chunking text...")
    chunks = split_text(text, CHUNK_SIZE)

    print("[3] Embedding...")
    embeddings, model = embed_chunks(chunks, EMBED_MODEL)

    print("[4] Indexing with FAISS...")
    index = create_faiss_index(embeddings)

    print("[5] Retrieving top matches...")
    retrieved_chunks = retrieve_chunks(query, chunks, model, index, TOP_K)

    print("[6] Generating response using OpenAI GPT...")
    return generate_response(query, retrieved_chunks)

# ============ Entry Point ============
if __name__ == "__main__":
    try:
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF path '{PDF_PATH}' is invalid.")
        
        user_question = input("Enter your question: ").strip()
        if not user_question:
            raise ValueError("Question cannot be empty.")
        
        answer = rag_pipeline(PDF_PATH, user_question)
        print("\n💬 Answer:\n", answer)
    
    except Exception as e:
        print(f"[ERROR] {e}")
