import time
import numpy as np
import psutil
import os
import csv

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


def process_pdf(pdf_path, chunk_size=300, overlap=50):
    """
    Extract text from the PDF, preprocess it, split into chunks,
    generate embeddings, and return a list of doc_entries.
    """
    doc_entries = []

    text_by_page = extract_text_from_pdf(pdf_path)
   
    for page_num, text in text_by_page:

        processed_text = preprocess_text(
            text,
            remove_whitespace=False,
            remove_punctuation=False,
            remove_stopwords=False,
            lemmatize=False
        )


        chunks = split_text_into_chunks(processed_text, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)  # from ingest
            doc_entries.append({
                "id": f"page_{page_num}_chunk_{i}",
                "file": os.path.basename(pdf_path),
                "page": page_num,
                "chunk": chunk,
                "embedding": embedding
            })

    return doc_entries

def test_redis_timed(doc_entries, test_query):
    """
    Clears Redis, creates HNSW index, stores the doc_entries, 
    then runs a single or multiple queries. Returns (indexing_time_ms, query_time_ms).
    """

    from redis.commands.search.query import Query

    clear_redis_store()
    create_hnsw_index()

    start_index = time.time()
    for doc in doc_entries:
        store_embedding(
            doc["file"],
            str(doc["page"]),
            doc["id"],
            doc["embedding"]
        )
    indexing_time_ms = (time.time() - start_index) * 1000

    query_embedding = get_embedding(test_query)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    start_query = time.time()
    results = redis_client.ft("embedding_index").search(q, query_params={"vec": query_vector})
    query_time_ms = (time.time() - start_query) * 1000

    return indexing_time_ms, query_time_ms

def experiment_chunk_sizes(pdf_paths, chunk_sizes, overlap_sizes, test_query, runs=5):
    """
    Loop over each chunk_size and overlap_size combination,
    measure how long it takes to process the PDF (splitting + embedding),
    and then measure Redis indexing/query times.
    """

    results = []

    for cs in chunk_sizes:
        for ov in overlap_sizes:
            for run_idx in range(1, runs+1):
                print(f"\n=== Testing chunk_size={cs}, overlap={ov}, run={run_idx} ===")

                start_proc = time.time()
                all_doc_entries = []
                for pdf_path in pdf_paths:
                    entries_for_pdf = process_pdf(pdf_path, chunk_size=cs, overlap=ov)
                    all_doc_entries.extend(entries_for_pdf)

                proc_time_ms = (time.time() - start_proc) * 1000
                redis_indexing_ms, redis_query_ms = test_redis_timed(all_doc_entries, test_query)

                # Get memory usage
                mem_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

                # Print to console
                print(f"Processing time (ms): {proc_time_ms:.2f}")
                print(f"Redis indexing time (ms): {redis_indexing_ms:.2f}")
                print(f"Redis query time (ms): {redis_query_ms:.2f}")
                print(f"Memory usage (MB): {mem_usage_mb:.2f}")

                # Append this run’s data as one row
                results.append([
                    float(cs),           
                    float(ov),          
                    float(run_idx),
                    float(proc_time_ms), # PDF processing time
                    float(redis_indexing_ms),
                    float(redis_query_ms),
                    float(mem_usage_mb)
                ])

    # Save to csv
    csv_filename = "src/chunk_size.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            "chunk_size",
            "overlap",
            "run_number",
            "processing_time_ms",
            "redis_indexing_time_ms",
            "redis_query_time_ms",
            "memory_usage_mb"
        ])
        writer.writerows(results)

    print(f"\nAll experiments complete. Results written to {csv_filename}")


def main():
    pdf_paths = ["data/02 - The Relational Model and Rel Algebra - Instructor.pdf",
                 "data/03 - Moving Beyond the Relational Model.pdf",
                 "data/05 - NoSQL Intro + KV DBs.pdf"]

    chunk_sizes = [200, 500, 1000, 2000]
    overlap_sizes = [0, 50, 100]

 
    test_query = "What is relational algebra?"

    # Run experiment
    experiment_chunk_sizes(pdf_paths, chunk_sizes, overlap_sizes, test_query)

if __name__ == "__main__":
    main()
