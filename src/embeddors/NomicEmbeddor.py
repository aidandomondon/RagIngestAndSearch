from embeddors.Embeddor import Embeddor

class NomicEmbeddor(Embeddor):

    def get_dimension(self) -> int:
        return 768

    @property
    def model(self) -> str:
        return "nomic-embed-text"

    def get_embedding(self, chunk: str) -> list:
        return super().get_embedding(chunk)