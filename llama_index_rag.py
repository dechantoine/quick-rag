# -*- coding: utf-8 -*-
import os
from loguru import logger

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

from typing import Iterator

from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.llms import (
    LLMMetadata,
)
from llama_index.types import PydanticProgramMode
from llama_index.prompts import PromptTemplate
from llama_index.response.schema import StreamingResponse

from llama_cpp import Llama
from llama_cpp.llama_types import CompletionChunk

from prompts import SYSTEM_PROMPT_MISTRAL, QUERY_WRAPPER_PROMPT_MISTRAL


class MyLocalRAG:

    @logger.catch
    def __init__(self):

        class LocalEmbeddingModel:
            def __init__(self) -> None:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            def get_text_embedding_batch(self, texts, **kwargs) -> list[list[float]]:
                return self.model.encode(texts,
                                         convert_to_numpy=True).tolist()

            def get_agg_embedding_from_queries(self, queries, **kwargs) -> list[float]:
                return self.model.encode(queries,
                                         convert_to_numpy=True).tolist()


        class LocalLLM():

            context_window: int = 32768
            num_output: int = 256
            model_name: str = "custom"

            def __init__(self) -> None:
                model_paths = LLM_MODEL_NAME.split("/")
                self.system_prompt = PromptTemplate(SYSTEM_PROMPT_MISTRAL)
                self.query_wrapper_prompt = PromptTemplate(QUERY_WRAPPER_PROMPT_MISTRAL)
                self.pydantic_program_mode = PydanticProgramMode.DEFAULT

                if not os.path.exists("temp"):
                    os.makedirs("temp")

                if not os.path.exists(os.path.join("temp", model_paths[2])):
                    hf_hub_download(repo_id=model_paths[0] + "/" + model_paths[1],
                                    filename=model_paths[2],
                                    repo_type="model",
                                    local_dir="temp",
                                    local_dir_use_symlinks=False)

                self.model = Llama(
                    model_path=os.path.join("temp", model_paths[2]),
                    n_ctx=2048,
                    # The max sequence length to use - note that longer sequence lengths require much more resources
                    n_threads=7,
                    # The number of CPU threads to use, tailor to your system and the resulting performance
                    n_gpu_layers=0,
                    # The number of layers to offload to GPU, if you have GPU acceleration available
                    # Set to 0 if no GPU acceleration is available on your system.
                )

            def format_query(self, query, **kwargs) -> str:
                return (self.system_prompt.get_template()
                        + self.query_wrapper_prompt.format(query_str=query.format_messages(**kwargs)[0].content))

            def predict(self, query, **kwargs) -> str:
                formatted_query = self.format_query(query, **kwargs)
                logger.info(f"query: {formatted_query}")
                response = self.model(formatted_query,
                                      max_tokens=1024,
                                      stop=["</s>"],
                                      echo=False,)
                logger.info(f"response: {response}")
                return response["choices"][0]["text"]

            def stream(self, query, **kwargs) -> Iterator[CompletionChunk]:
                formatted_query = self.format_query(query, **kwargs)
                logger.info(f"query: {formatted_query}")
                response = self.model(formatted_query,
                                      max_tokens=1024,
                                      stop=["</s>"],
                                      stream=True,
                                      echo=False)
                logger.info(f"response: {response}")
                for part in response:
                    chunk = part["choices"][0]["text"]
                    yield chunk

            @property
            def metadata(self) -> LLMMetadata:
                """Get LLM metadata."""
                return LLMMetadata(
                    context_window=self.context_window,
                    num_output=self.num_output,
                    model_name=self.model_name,
                )

        DATA_DIR = os.environ.get("DATA_DIR", "data")
        EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
        LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME",
                                        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q5_K_M.gguf")

        embedding_model = LocalEmbeddingModel()
        llm = LocalLLM()

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
            # refine_template=refine_template,
            # response_mode="no_text",
        )

    @logger.catch
    async def query(self, message: str):
        logger.info(message)
        response = self.query_engine.query(message)
        for tkn in response.response_gen:
            yield tkn