## DS 4300 Example - from docs

import ollama
import numpy as np
import chromadb
import os
import fitz

# Initialize Chroma connection
client_settings = chromadb.Settings()
client_settings.allow_isreset = True
chroma_client = chromadb.HttpClient(host="localhost", port=8000, settings=client_settings)

# Initialize Chroma wrapper for custom embedding model
class ChromaCustomEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return chromadb.Embeddings(map(lambda doc: get_embedding(doc), input))
embedding_function = ChromaCustomEmbeddingFunction()

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the chroma vector store
def clear_chroma_store():
    print("Clearing existing Chroma store...")
    if chroma_client.reset():
        print("Chroma store cleared.")
    else:
        raise Exception("Chroma store unable to be cleared.")


# Create a collection in Chroma
def create_hnsw_index():
    try:
        chroma_client.delete_collection(INDEX_NAME)
    except Exception as e:
        raise Exception(e)
    chroma_client.create_collection(
        name=INDEX_NAME,
        metadata=chromadb.CollectionMetadata({
            "hsnw:space": DISTANCE_METRIC.lower()
        }),
        embedding_function=embedding_function
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Chroma
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    collection = chroma_client.get_collection(INDEX_NAME, embedding_function)
    collection.add(
        ids=[key],
        embeddings=[embedding],
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_chroma(query_text: str):
    collection = chroma_client.get_collection(INDEX_NAME, embedding_function)
    res = collection.query(
        query_texts=query_text,
        n_results=5,
        include="distances"
    )
    for i in range(len(res["ids"])):
        id = res["ids"][i]
        distance = res["distances"][i]
        print(f"{id} \n ----> {distance}\n")


def main():
    clear_chroma_store()
    create_hnsw_index()

    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")


if __name__ == "__main__":
    main()
