from datetime import datetime
from typing import Tuple

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnableMap


def get_openai_type(msg):
    if msg.type == "human":
        return "user"
    if msg.type == "ai":
        return "assistant"
    if msg.type == "chat":
        return msg.role
    return msg.type


def get_chain(
    system_prompt: str, temperature: float = 0.7
) -> Tuple[Runnable, ConversationBufferMemory]:
    """Return a chain defined primarily in LangChain Expression Language"""
    memory = ConversationBufferMemory(return_messages=True)
    ingress = RunnableMap(
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: memory.load_memory_variables(x)["history"],
            "time": lambda _: str(datetime.now()),
        }
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "\nIt's currently {time}.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(temperature=temperature)
    chain = ingress | prompt | llm
    return chain, memory
