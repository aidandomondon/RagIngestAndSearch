from tqdm import tqdm
import fitz
import os
from textpreparers.TextPreparer import TextPreparer
from chunkers.Chunker import Chunker
from embeddors.Embeddor import Embeddor
from vectordbs.VectorDB import VectorDB

class Ingestor():

    def __init__(self, 
        text_preparer: TextPreparer, 
        chunker: Chunker,
        embeddor: Embeddor, 
        vector_db: VectorDB
    ):
        self.text_preparer = text_preparer
        self.chunker = chunker
        self.embeddor = embeddor
        self.vector_db = vector_db

    # extract the text from a PDF by page
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text_by_page = []
        for page_num, page in enumerate(doc):
            text_by_page.append((page_num, page.get_text()))
        return text_by_page

    # extract the text from a .txt or .ipynb file
    def _extract_text_from_txt(self, txt_path):
        """Extract text from  a txt file"""
        text = open(txt_path, "r").read()
        return [(0, text)]
    

    # Process all files in a given directory
    def ingest(self, data_dir):

        for file_name in tqdm(os.listdir(data_dir), unit='file'):

            file_path = os.path.join(data_dir, file_name)

            if file_name.endswith(".pdf"):
                text_by_page = self._extract_text_from_pdf(file_path)
            elif file_name.endswith(".txt") or file_name.endswith(".ipynb"):
                text_by_page = self._extract_text_from_txt(file_path)
            else:
                continue # skip unsupported file types

            for page_num, text in text_by_page:
                text = self.text_preparer.preprocess_text(text)
                chunks = self.chunker.split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = self.embeddor.get_embedding(chunk)
                    self.vector_db.store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
