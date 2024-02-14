import os

from llama_index.llms import (
    LLMMetadata,
)
from llama_index.types import PydanticProgramMode
from llama_index.prompts import PromptTemplate
from llama_index.response.schema import StreamingResponse

from llama_cpp import Llama
from llama_cpp.llama_types import CompletionChunk

from prompts import SYSTEM_PROMPT_MISTRAL, QUERY_WRAPPER_PROMPT_MISTRAL

from huggingface_hub import hf_hub_download

from typing import Iterator

from loguru import logger


class LocalLLM():
    context_window: int = 32768
    num_output: int = 256
    model_name: str = "custom"

    def __init__(self, llm_model_name) -> None:
        model_paths = llm_model_name.split("/")
        self.system_prompt = PromptTemplate(SYSTEM_PROMPT_MISTRAL)
        self.query_wrapper_prompt = PromptTemplate(QUERY_WRAPPER_PROMPT_MISTRAL)
        self.pydantic_program_mode = PydanticProgramMode.DEFAULT

        self.download_model(model_paths)
        self.load_model(model_paths[2])

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

    def download_model(self, model_paths: list[str]) -> None:
        if not os.path.exists("temp"):
            os.makedirs("temp")

        if not os.path.exists(os.path.join("temp", model_paths[2])):
            hf_hub_download(repo_id=model_paths[0] + "/" + model_paths[1],
                            filename=model_paths[2],
                            repo_type="model",
                            local_dir="temp",
                            local_dir_use_symlinks=False)

    def load_model(self, model_name: str) -> None:
        self.model = Llama(
            model_path=os.path.join("temp", model_name),
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
                              echo=False, )
        logger.info(f"response: {response}")
        return response["choices"][0]["text"]

    def stream(self, query, **kwargs) -> Iterator[str]:
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
