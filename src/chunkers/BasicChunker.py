from chunkers.Chunker import Chunker
from typing import List

class BasicChunker(Chunker):

    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # split the text into chunks with overlap
    def split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks of approximately chunk_size words with overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks