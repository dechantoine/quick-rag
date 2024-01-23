# -*- coding: utf-8 -*-
import os
from loguru import logger

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.schema import NodeWithScore
from llama_cpp import Llama

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
DATA_DIR = os.environ.get("DATA_DIR", "data")

class MyLocalRAG:
    def __init__(self):

        class LocalEmbeddingModel:
            def __init__(self):
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            def get_text_embedding_batch(self, texts, **kwargs):
                return self.model.encode(texts,
                                         convert_to_numpy=True).tolist()

            def get_agg_embedding_from_queries(self, queries, **kwargs):
                return self.model.encode(queries,
                                         convert_to_numpy=True).tolist()


        class LocalLLM:
            def __init__(self):
                model_paths = LLM_MODEL_NAME.split("/")

                if not os.path.exists("temp"):
                    os.makedirs("temp")

                if not os.path.exists(os.path.join("temp", model_paths[2])):
                    hf_hub_download(repo_id=model_paths[0] + model_paths[1],
                                    filename=model_paths[2],
                                    repo_type="model",
                                    local_dir="temp",
                                    local_dir_use_symlinks=False)

                self.model = Llama(
                    model_path=os.path.join("temp", model_paths[2]),
                    n_ctx=512,
                    # The max sequence length to use - note that longer sequence lengths require much more resources
                    n_threads=8,
                    # The number of CPU threads to use, tailor to your system and the resulting performance
                    n_gpu_layers=0
                    # The number of layers to offload to GPU, if you have GPU acceleration available
                    # Set to 0 if no GPU acceleration is available on your system.
                )

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
            # rebuild index from data by calling embedding model
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
