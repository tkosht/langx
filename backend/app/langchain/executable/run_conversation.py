
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

if __name__ == "__main__":
    memory = ConversationBufferMemory(return_messages=True)