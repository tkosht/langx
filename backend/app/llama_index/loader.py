from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore


def load_index(data_dir_="data/llama_faiss") -> VectorStoreIndex:
    embed_model_name = "intfloat/multilingual-e5-small"
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))
    Settings.embed_model = embed_model
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    vector_store = FaissVectorStore.from_persist_dir(data_dir_)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=data_dir_)
    index: VectorStoreIndex = load_index_from_storage(storage_context=storage_context)
    return index
