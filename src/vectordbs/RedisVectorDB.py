from vectordbs.VectorDB import VectorDB
from redis import Redis
from redis.exceptions import ResponseError
from redis.commands.search.query import Query
from embeddors.Embeddor import Embeddor
import numpy as np

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

class RedisVectorDB(VectorDB):

    def __init__(self, dimension: int):
        super().__init__(dimension)
        self.client = Redis(host="localhost", port=6379, db=0)
        self._clear_store()

    def clean_and_reinit(self, dimension: int) -> None:
        # Clear vector db store
        self._clear_store()
        # Set the vector db to deal with vectors of the dimension
        # outputted by the embeddor
        self.dimension = dimension
        self._create_hnsw_index()

    # Clear the Redis store
    def _clear_store(self):
        print("Clearing existing Redis store...")
        self.client.flushdb()
        print("Redis store cleared.")

    # Create an HNSW index in Redis
    def _create_hnsw_index(self):
        try:
            self.client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
        except ResponseError:
            pass

        self.client.execute_command(
            f"""
            FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.dimension} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
            """
        )
        # FT.CREATE "embedding_index" ON HASH PREFIX 1 "doc:" SCHEMA text TEXT embedding VECTOR HNSW 6 DIM 768 TYPE FLOAT32 DISTANCE_METRIC "COSINE"
        print("Index created successfully.")
        
    # Store the embedding in Redis
    def store_embedding(self, file: str, page: int, chunk: str, embedding: list):
        key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
        self.client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(
                    embedding, dtype=np.float32
                ).tobytes(),  # Store as byte array
            },
        )
    
    def search_embeddings(self, query_embedding, top_k=3) -> list:
        """
        Search for the given embedding in Redis.
        """

        # Convert embedding to bytes for Redis search
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        try:
            # Construct the vector similarity search query
            # Use a more standard RediSearch vector search syntax
            # q = Query("*").sort_by("embedding", query_vector)

            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "file", "page", "chunk", "vector_distance")
                .dialect(2)
            )

            # Perform the search
            results = self.client.ft(INDEX_NAME).search(
                q, query_params={"vec": query_vector}
            )

            # Transform results into the expected format
            top_results = [
                {
                    "file": result.file,
                    "page": result.page,
                    "chunk": result.chunk,
                    "similarity": result.vector_distance,
                }
                for result in results.docs
            ][:top_k]

            return top_results

        except Exception as e:
            print(f"Search error: {e}")
            return []
        
    def memory_usage(self) -> float:
        return self.client.ft(INDEX_NAME).info()['vector_index_sz_mb']