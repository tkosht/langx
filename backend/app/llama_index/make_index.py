import glob
from inspect import signature

import faiss
import typer
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from omegaconf import DictConfig

# from transformers import AutoTokenizer, AutoModel

to_dim: dict = {
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-small": 384,
}


def create_faiss_index(embed_model_name: str, documents: list):
    d = to_dim[embed_model_name]
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index


def _main(params: DictConfig):
    # return index.search(params.query)
    embed_model_name = "intfloat/multilingual-e5-small"
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=embed_model_name))
    Settings.embed_model = embed_model
    Settings.text_splitter = SentenceSplitter(chunk_size=1024)

    pdf_files = sorted(glob.glob("data/pdf/*.pdf"))
    documents = []
    for pdf in pdf_files:
        docs = SimpleDirectoryReader(input_files=[pdf]).load_data()
        for idx, doc in enumerate(docs):
            doc.metadata["file"] = pdf
            doc.metadata["page"] = idx + 1
        documents.extend(docs)

    index = create_faiss_index(embed_model_name, documents)
    index.storage_context.persist("data/llama_faiss")


def config():
    cfg = DictConfig(dict(is_experiment=True, do_share=False))
    return cfg


def main(
    do_share: bool = None,
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    typer.run(main)
