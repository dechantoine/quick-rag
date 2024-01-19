import chainlit as cl
from loguru import logger
from rag import MyLocalRAG

@cl.on_chat_start
async def on_chat_start() -> None:
    """This function is called when the chat is started.

    It is used to initialize the chatbot.

    """
    rag = MyLocalRAG()

    cl.user_session.set("rag", rag)


@cl.on_message
async def main(message: cl.Message):
    rag = cl.user_session.get("rag")
    logger.info(message.content)

    #response = await cl.make_async(rag.query)(message.content)
    response = rag.query(message.content)

    logger.info(response)

    response_message = cl.Message(content=response)

    await response_message.send()
