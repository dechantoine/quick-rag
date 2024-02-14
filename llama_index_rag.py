# -*- coding: utf-8 -*-
import os
from loguru import logger

from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context

from llama_llm import LocalLLM
from sentence_transformers_embedder import LocalEmbeddingModel


class MyLocalRAG:

    @logger.catch
    def __init__(self) -> None:

        DATA_DIR = os.environ.get("DATA_DIR", "data")
        EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
        LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME",
                                        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q5_K_M.gguf")

        embedding_model = LocalEmbeddingModel(embedding_model_name=EMBEDDING_MODEL_NAME)
        llm = LocalLLM(llm_model_name=LLM_MODEL_NAME)

        service_context = ServiceContext.from_defaults(
            embed_model=embedding_model,
            llm=llm,
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
            # rebuild index from data by calling embedding model
            from llama_index import VectorStoreIndex, SimpleDirectoryReader

            logger.info("Building index from data in folder {}...".format(DATA_DIR))

            documents = SimpleDirectoryReader(DATA_DIR).load_data()
            index = VectorStoreIndex.from_documents(documents=documents,
                                                    show_progress=True)
            index.storage_context.persist(persist_dir="storage")

        self.query_engine = index.as_query_engine(
            streaming=True,
            service_context=service_context,
        )

    @logger.catch
    async def query(self, message: str) -> str:
        logger.info(message)
        response = self.query_engine.query(message)
        for tkn in response.response_gen:
            yield tkn
