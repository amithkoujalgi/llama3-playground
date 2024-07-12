from sentence_transformers import SentenceTransformer
from typing import List


class MyEmbeddings:
    def __init__(self, model: str):
        self.model = SentenceTransformer(model, trust_remote_code=True, device="cuda")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        # return self.model.encode([query])
        return self.model.encode(query).tolist()


def get_embedding_function(embed_model_Path: str):
    embedding_function = MyEmbeddings(embed_model_Path)
    return embedding_function
