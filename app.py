import os
import chainlit as cl
from loguru import logger
from prompts import WELCOME, ASK_FOLDER, LOADING_DONE
from llama_index_rag import MyLocalRAG

TEMPLATE_RESPONSE = "Bonjour, j'ai trouvé {} résultats pour votre recherche : \n\n{}"
TEMPLATE_NODE = "{}. (score: {:.2f})\n{}\n\n"

@cl.on_chat_start
@logger.catch
async def on_chat_start() -> None:
    """This function is called when the chat is started.

    It is used to initialize the chatbot.

    """
    res = await cl.AskUserMessage(content=WELCOME + ASK_FOLDER).send()
    if res:
        #TODO: check if the folder exists + display n first docs + ask for validation (button)
        os.environ["DATA_DIR"] = res['output']

    async_local_rag = cl.make_async(MyLocalRAG)
    rag = await async_local_rag()

    cl.user_session.set("rag", rag)

    done = cl.Message(content=LOADING_DONE)
    await done.send()


@cl.on_message
@logger.catch
async def main(message: cl.Message):
    rag = cl.user_session.get("rag")

    streamed_answer = await cl.make_async(rag.query)(message.content)

    #formatted_sources = "".join([TEMPLATE_NODE.format(i + 1, node.score, node.node.get_content())
    #                             for i, node in enumerate(response)])
    #formatted_response = TEMPLATE_RESPONSE.format(len(response), formatted_sources)

    response_message = cl.Message(content="")
    async for chunk in streamed_answer:
        if token := chunk or "":
            await response_message.stream_token(token)

    await response_message.update()
