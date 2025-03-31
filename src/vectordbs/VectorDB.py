from abc import ABC, abstractmethod
from embeddors.Embeddor import Embeddor

class VectorDB(ABC):

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    def clean_and_reinit(self, dimension: int) -> None:
        """
        Clear database contents and reinitialize for intake.
        """
        ...

    @abstractmethod
    def store_embedding(self, file: str, page: int, chunk: str, embedding: list) -> None:
        """
        Ingests the given chunk, along with its meta data, into the vector database.
        """
    
    @abstractmethod
    def search_embeddings(self, query_embedding, top_k=3) -> list:
        """
        Search for the given embedding.
        """
        ...

    @abstractmethod
    def memory_usage(self) -> float:
        """
        Returns the memory usage (in mb) of the index containing the documents.
        """
        ...