from embeddors.Embeddor import Embeddor

class MiniLMEmbeddor(Embeddor):

    def get_dimension(self) -> int:
        return 384

    @property
    def model(self) -> str:
        return "all-minilm"

    def get_embedding(self, chunk: str) -> list:
        return super().get_embedding(chunk)