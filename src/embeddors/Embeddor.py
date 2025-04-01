from abc import ABC, abstractmethod
from ollama import embeddings

class Embeddor(ABC):
    """
    Embeds chunks into a vector space.
    """

    @abstractmethod
    def get_dimension(self) -> int:
        """Returns the dimension of vectors in this vector space."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        pass

    def get_embedding(self, chunk: str) -> list:
        """
        Return the embedding of the given chunk.
        """
        response = embeddings(model=self.model, prompt=chunk)
        return response["embedding"]