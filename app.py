import os
import chainlit as cl
from loguru import logger
from prompts import WELCOME, ASK_FOLDER
from rag import MyLocalRAG

TEMPLATE_RESPONSE = "Bonjour, j'ai trouvé {} résultats pour votre recherche : \n\n{}"
TEMPLATE_NODE = "{}. (score: {:.2f})\n{}\n\n"

@cl.on_chat_start
async def on_chat_start() -> None:
    """This function is called when the chat is started.

    It is used to initialize the chatbot.

    """
    res = await cl.AskFileMessage(content=WELCOME + ASK_FOLDER).send()
    if res:
        os.environ["DATA_DIR"] = res['output']

    rag = MyLocalRAG()

    cl.user_session.set("rag", rag)


@cl.on_message
async def main(message: cl.Message):
    rag = cl.user_session.get("rag")

    response = rag.query(message.content)

    formatted_sources = "".join([TEMPLATE_NODE.format(i + 1, node.score, node.node.get_content())
                                 for i, node in enumerate(response)])
    formatted_response = TEMPLATE_RESPONSE.format(len(response), formatted_sources)

    response_message = cl.Message(content=formatted_response)
    await response_message.send()
