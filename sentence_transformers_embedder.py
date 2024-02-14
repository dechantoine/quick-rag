from sentence_transformers import SentenceTransformer

class LocalEmbeddingModel:
    def __init__(self, embedding_model_name) -> None:
        self.model = SentenceTransformer(embedding_model_name)

    def get_text_embedding_batch(self, texts, **kwargs) -> list[list[float]]:
        return self.model.encode(texts,
                                 convert_to_numpy=True).tolist()

    def get_agg_embedding_from_queries(self, queries, **kwargs) -> list[float]:
        return self.model.encode(queries,
                                 convert_to_numpy=True).tolist()