import os
import shutil
import re
from typing import List

# import pysqlite3
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from rag_utils import get_embedding_function


class CreateChromaDB:
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 100
    TOP_K = 5

    def __init__(self, db_path: str, embed_model_path: str, clear_db: bool = False):
        self.chroma_path = f"{db_path}/chroma"
        self.clear_db = clear_db
        self.embedding_function = get_embedding_function(embed_model_path)

    def populate_database(self, ocr_text: str, pdf_file_path: str):
        if self.clear_db:
            print("Clearing Database")
            self.clear_database()
        if not os.path.exists(self.chroma_path):
            chunks = self.generate_chunks_from_ocr_text(ocr_text=ocr_text, pdf_file_path=pdf_file_path)
            self.add_to_chroma(chunks=chunks)
        else:
            print("Database already created!!!!!")

    def generate_chunks_from_ocr_text(self, ocr_text: str, pdf_file_path: str) -> List[Document]:
        split_pages = re.split(r'---PAGE \d+---', ocr_text)
        split_pages = [page.strip() for page in split_pages if page.strip()]
        chunks = []
        for idx, page in enumerate(split_pages):
            doc = Document(page_content=page, metadata={"source": pdf_file_path, "page": idx})
            chunks.append(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CreateChromaDB.CHUNK_SIZE,
                                                       chunk_overlap=CreateChromaDB.CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(chunks)
        return chunks

    def add_to_chroma(self, chunks):
        db = Chroma(
            persist_directory=self.chroma_path, embedding_function=self.embedding_function
        )

        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        chunks_with_ids = self.calculate_chunk_ids(chunks=chunks)
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("No new documents to add")

    def calculate_chunk_ids(self, chunks) -> List[Document]:
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks

    def clear_database(self):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)

    def retrieve_relevant_chunks(self, query_text: str) -> str:
        db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_function)
        results = db.similarity_search_with_score(query_text, k=CreateChromaDB.TOP_K)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        print(f"context_text: \n{context_text}\n")
        return context_text
