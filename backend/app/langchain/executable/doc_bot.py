import glob
from inspect import signature

import gradio as gr
import typer
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA  # , VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader  # , TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from omegaconf import DictConfig


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def _bot(history, prompt, qa):
    query = history[-1][0]
    query = prompt.format(question=query)
    answer = qa.run(query)
    source = qa._get_docs(query)[0]
    source_sentence = source.page_content
    answer_source = "```" + source_sentence + "```" + "\n" \
                    + "source:" + source.metadata["source"] + ", page:" + str(source.metadata["page"])
    history[-1][1] = answer + "\n\n情報ソースは以下です：\n" + answer_source
    return history


def _main(params: DictConfig):
    # 検索対象のドキュメントを読み込む
    documents = []
    pdf_list = glob.glob("data/pdf/*.pdf")
    for pdf in pdf_list:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        documents += docs

    # テキストをベクトル化
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
                                     chain_type="stuff",
                                     retriever=vectordb.as_retriever(),
                                     verbose=True)

    template = """
    あなたは親切なアシスタントです。下記の質問に日本語で回答してください。
    ただし回答を考える際には step-by-step で丁寧に検討をしてください。
    質問：{question}
    回答：
    """

    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )

    def bot(history):
        return _bot(history, prompt, qa)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot([], elem_id="demobot").style(height=400)

        with gr.Row():
            with gr.Column(scale=0.3):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="質問を入れてちょ✨",
                ).style(container=False)

        txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
            bot, chatbot, chatbot
        )

    if params.do_share:
        demo.launch(share=True, auth=("user", "user123"), server_name="0.0.0.0", server_port=7860)
    else:
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


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
    load_dotenv()

    typer.run(main)
