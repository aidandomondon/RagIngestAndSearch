from embeddors.Embeddor import Embeddor

class MiniLM33Embeddor(Embeddor):

    def get_dimension(self) -> int:
        return 384

    @property
    def model(self) -> str:
        return "all-minilm:33m"

    def get_embedding(self, chunk: str) -> list:
        return super().get_embedding(chunk)