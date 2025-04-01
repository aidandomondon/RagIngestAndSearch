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

def process_pdfs(pdf_paths,
                 remove_whitespace=False,
                 remove_punctuation=False,
                 remove_stopwords=False,
                 chunk_size=300,
                 overlap=50):
    """
    Loop over each PDF in pdf_paths, extract+preprocess+split+embed.
    Return a single list of doc_entries for all PDFs combined.
    """
    all_doc_entries = []
    for pdf_path in pdf_paths:
        text_by_page = extract_text_from_pdf(pdf_path)

        for page_num, text in text_by_page:
            # Preprocess with the chosen flags
            processed_text = preprocess_text(
                text,
                remove_whitespace=remove_whitespace,
                remove_punctuation=remove_punctuation,
                remove_stopwords=remove_stopwords,
                lemmatize=False
            )

            # Split into chunks
            chunks = split_text_into_chunks(processed_text, chunk_size, overlap)

            # Embed chunks
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                doc_entries = {
                    "id": f"{os.path.basename(pdf_path)}_p{page_num}_c{i}",
                    "file": os.path.basename(pdf_path),
                    "page": page_num,
                    "chunk": chunk,
                    "embedding": embedding
                }
                all_doc_entries.append(doc_entries)

    return all_doc_entries


def test_redis_timed(doc_entries, test_query):
    """
    Clears Redis, creates HNSW index, stores doc_entries,
    queries them, and returns (indexing_time_ms, query_time_ms).
    """
    from redis.commands.search.query import Query

    clear_redis_store()
    create_hnsw_index()

    # Indexing
    start_index = time.time()
    for doc in doc_entries:
        store_embedding(doc["file"], str(doc["page"]), doc["id"], doc["embedding"])
    indexing_time_ms = (time.time() - start_index) * 1000

    # Query
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

def main():

    pdf_paths = ["data/02 - The Relational Model and Rel Algebra - Instructor.pdf",
                "data/03 - Moving Beyond the Relational Model.pdf",
                "data/05 - NoSQL Intro + KV DBs.pdf"]

    chunk_size = 300
    overlap = 50

    test_query = "What is relational algebra?"

    runs_per_scenario = 5

    preprocessing_scenarios = [
        ("NoPreproc", False, False, False),
        ("WhitespaceOnly", True,  False, False),
        ("PunctuationOnly",False, True,  False),
        ("StopwordsOnly",  False, False, True),
        ("AllTrue",       True,  True,  True),
    ]

    results = []

    for scenario_name, rm_ws, rm_punct, rm_stop in preprocessing_scenarios:
        for run_idx in range(1, runs_per_scenario + 1):
            print(f"\n=== Testing {scenario_name}, run {run_idx} ===")

            start_proc = time.time()
            doc_entries = process_pdfs(
                pdf_paths,
                remove_whitespace=rm_ws,
                remove_punctuation=rm_punct,
                remove_stopwords=rm_stop,
                chunk_size=chunk_size,
                overlap=overlap
            )
            proc_time_ms = (time.time() - start_proc) * 1000

            redis_indexing_ms, redis_query_ms = test_redis_timed(doc_entries, test_query)

            mem_usage_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            print(f"Scenario: {scenario_name}, Run {run_idx}")
            print(f"Processing time (ms): {proc_time_ms:.2f}")
            print(f"Redis indexing time (ms): {redis_indexing_ms:.2f}")
            print(f"Redis query time (ms): {redis_query_ms:.2f}")
            print(f"Memory usage (MB): {mem_usage_mb:.2f}")

            results.append([
                scenario_name,
                rm_ws, rm_punct, rm_stop,
                run_idx,
                proc_time_ms,
                redis_indexing_ms,
                redis_query_ms,
                mem_usage_mb
            ])

    # Write to CSV
    csv_filename = "src/preprocessing_test_results.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_name",
            "remove_whitespace",
            "remove_punctuation",
            "remove_stopwords",
            "run_number",
            "processing_time_ms",
            "redis_indexing_time_ms",
            "redis_query_time_ms",
            "memory_usage_mb"
        ])
        writer.writerows(results)

    print(f"\nAll preprocessing scenarios complete. Results saved to {csv_filename}")


if __name__ == "__main__":
    main()
