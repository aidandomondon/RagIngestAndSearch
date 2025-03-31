from abc import ABC, abstractmethod
from typing import List

class Chunker(ABC):

    @abstractmethod
    def split_text_into_chunks(self, text: str) -> List[str]:
        ...