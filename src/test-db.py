import os
import time
import numpy as np
import psutil
import uuid

from ingest import (
    extract_text_from_pdf,
    preprocess_text,
    split_text_into_chunks,
    get_embedding,
    clear_redis_store,
    create_hnsw_index,
    store_embedding,
    redis_client
)

TEST_QUERY = "What is relational algebra?"

def process_pdf(pdf_path, chunk_size=300, overlap=50):
    doc_entries = []
    text_by_page = extract_text_from_pdf(pdf_path)
    for page_num, text in text_by_page:
        processed_text = preprocess_text(text, remove_whitespace=True, remove_punctuation=True)
        chunks = split_text_into_chunks(processed_text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            entry = {
                "id": f"{os.path.basename(pdf_path)}_p{page_num}_c{i}",
                "file": os.path.basename(pdf_path),
                "page": page_num,
                "chunk": chunk,
                "embedding": embedding
            }
            doc_entries.append(entry)
    return doc_entries

def print_memory_usage(db_name):
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"{db_name} memory usage: {mem_usage:.2f} MB")

def test_redis(doc_entries):
    from redis.commands.search.query import Query
    print("\n[Redis Test]")
    clear_redis_store()
    create_hnsw_index()
    
    start = time.time()
    for doc in doc_entries:
        store_embedding(doc["file"], str(doc["page"]), doc["id"], doc["embedding"])
    redis_indexing_time = (time.time() - start) * 1000  # ms
    
    query_embedding = get_embedding(TEST_QUERY)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
    q = (Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
         .sort_by("vector_distance")
         .return_fields("id", "vector_distance")
         .dialect(2))
    
    start = time.time()
    results = redis_client.ft("embedding_index").search(q, query_params={"vec": query_vector})
    redis_query_time = (time.time() - start) * 1000

    print(f"Test Query: {TEST_QUERY}")
    print(f"Redis indexing time: {redis_indexing_time:.2f} ms")
    print(f"Redis query time: {redis_query_time:.2f} ms")
    print(f"Redis query results count: {len(results.docs)}")
    print_memory_usage("Redis")

def test_chroma(doc_entries):
    import chromadb
    print("\n[Chroma Test]")
    client = chromadb.Client()
    try:
        client.delete_collection("ds4300_notes")
    except Exception:
        pass
    collection = client.create_collection(name="ds4300_notes")
    
    start = time.time()
    for doc in doc_entries:
        collection.add(
            documents=[doc["chunk"]],
            metadatas=[{"file": doc["file"], "page": doc["page"]}],
            ids=[doc["id"]],
            embeddings=[doc["embedding"]]
        )
    chroma_indexing_time = (time.time() - start) * 1000
    
    query_embedding = get_embedding(TEST_QUERY)
    start = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    chroma_query_time = (time.time() - start) * 1000
    
    num_results = len(results.get("ids")[0]) if results.get("ids") else 0
    print(f"Test Query: {TEST_QUERY}")
    print(f"Chroma indexing time: {chroma_indexing_time:.2f} ms")
    print(f"Chroma query time: {chroma_query_time:.2f} ms")
    print(f"Chroma query results count: {num_results}")
    print_memory_usage("Chroma")

def test_qdrant(doc_entries):
    from qdrant_client import QdrantClient
    print("\n[Qdrant Test]")
    
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "ds4300_collection"
    
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception as e:
        pass
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={"size": 768, "distance": "Cosine"}
    )
    
    points = []
    for doc in doc_entries:
        point = {
            "id": str(uuid.uuid4()),
            "vector": doc["embedding"],
            "payload": {"file": doc["file"], "page": doc["page"], "chunk": doc["chunk"]}
        }
        points.append(point)
    
    start = time.time()
    client.upsert(collection_name=collection_name, points=points)
    qdrant_indexing_time = (time.time() - start) * 1000
    
    query_embedding = get_embedding(TEST_QUERY)
    start = time.time()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=5
    )
    qdrant_query_time = (time.time() - start) * 1000
    print(f"Test Query: {TEST_QUERY}")
    print(f"Qdrant indexing time: {qdrant_indexing_time:.2f} ms")
    print(f"Qdrant query time: {qdrant_query_time:.2f} ms")
    print(f"Qdrant query results count: {len(search_result)}")
    print_memory_usage("Qdrant")

def main():
    pdf_path = os.path.join("..", "data", "02 - The Relational Model and Rel Algebra - Instructor.pdf")
    print(f"Processing PDF: {pdf_path}")
    doc_entries = process_pdf(pdf_path)
    print(f"Total document entries: {len(doc_entries)}")
    
    test_redis(doc_entries)
    test_chroma(doc_entries)
    test_qdrant(doc_entries)
    
    overall_mem = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)
    print(f"\nOverall process memory usage: {overall_mem:.2f} MB")

if __name__ == "__main__":
    main()
