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
    #response = await cl.make_async(rag.query)(message.content)
    response = rag.query(message.content)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    if response.response_txt:
        response_message.content = response.response_txt

    await response_message.send()
