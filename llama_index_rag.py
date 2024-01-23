# -*- coding: utf-8 -*-
import os
from loguru import logger

from sentence_transformers import SentenceTransformer

from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.schema import NodeWithScore

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
DATA_DIR = os.environ.get("DATA_DIR", "data")

class MyLocalRAG:
    def __init__(self):

        class LocalEmbeddingModel:
            def __init__(self):
                self.model = SentenceTransformer(MODEL_NAME)

            def get_text_embedding_batch(self, texts, **kwargs):
                return self.model.encode(texts,
                                         convert_to_numpy=True).tolist()

            def get_agg_embedding_from_queries(self, queries, **kwargs):
                return self.model.encode(queries,
                                         convert_to_numpy=True).tolist()

        embedding_model = LocalEmbeddingModel()

        service_context = ServiceContext.from_defaults(
            embed_model=embedding_model,
            llm=None,
            chunk_size=256,
            num_output=5
        )

        set_global_service_context(service_context)

        try:
            # rebuild storage context from local
            storage_context = StorageContext.from_defaults(persist_dir="storage")
            # load index
            index = load_index_from_storage(storage_context)
        except:
            # rebuild index from data by calling llm model
            from llama_index import VectorStoreIndex, SimpleDirectoryReader

            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(documents=documents,
                                                    show_progress=True)
            index.storage_context.persist(persist_dir="storage")

        self.query_engine = index.as_query_engine(
            streaming=False,
            service_context=service_context,
            # refine_template=refine_template,
            # response_mode="no_text",
        )

    def query(self, message: str) -> list[NodeWithScore]:
        logger.info(message)
        response = self.query_engine.query(message)
        logger.info(response)
        return response.source_nodes
