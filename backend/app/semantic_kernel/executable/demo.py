import re
from inspect import signature

import gradio as gr
import typer
from omegaconf import DictConfig

from app.semantic_kernel.component.semantic_bot import SemanticBot


def _find_urls(text_contains_urls: str) -> list[str]:
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    urls = re.findall(url_pattern, text_contains_urls)
    return urls


class BotWrapper(object):
    def __init__(self, bot: SemanticBot, memory_chunk_size: int = 1024) -> None:
        self.bot = bot
        self.memory_chunk_size = memory_chunk_size

        self.processed = set()

    async def add_memory(self, text: str):
        import requests
        from bs4 import BeautifulSoup

        urls = _find_urls(text)

        successful_urls = []
        for url in urls:
            if url in self.processed:
                continue

            try:
                self.processed.add(url)

                response = requests.get(url)
                soup = BeautifulSoup(response.text, "lxml")

                for idx in range(0, len(soup.text), self.memory_chunk_size):
                    await self.bot.memory.append(
                        text=soup.text[idx : idx + self.memory_chunk_size],
                        additional_metadata=url,
                    )
                successful_urls.append(url)
            except Exception as e:
                print(e)
                continue

        if not successful_urls:
            return "no urls to be loaded."

        return "succeeded to load into the memory: \n" + "\n".join(successful_urls)


def _init(
    history: list[tuple[str, str]], text: str
) -> tuple[list[tuple[str, str]], str]:
    history = history + [(text, None)]
    return history, ""


def _init_session(status: dict) -> dict:
    import uuid

    if "session_id" in status:
        return status

    status["session_id"] = str(uuid.uuid4())
    status["bot"] = bot = SemanticBot()
    status["bot_wrapper"] = BotWrapper(bot=bot)
    return status


async def _init_memory(status: dict, text: str) -> dict:
    status = _init_session(status)
    bot_wrapper: BotWrapper = status["bot_wrapper"]
    result: str = await bot_wrapper.add_memory(text)
    return status, result


async def _chat(
    status: dict,
    history: list[tuple[str, str]],
    model_name: str,
    temperature_percent: int,  # in [0, 200]
    max_tokens: int = 1024,
) -> list[tuple[str, str]]:
    assert "bot" in status
    bot: SemanticBot = status["bot"]
    new_history = await bot.do_chat(
        history=history,
        model_name=model_name,
        temperature_percent=temperature_percent,
        max_tokens=max_tokens,
    )
    return status, new_history


def _main(params: DictConfig):
    default_query = "LLMについて教えて？"

    with gr.Blocks() as demo:
        status = gr.State({})

        with gr.Tab("Conversation"):
            with gr.Row():
                chatbot = gr.Chatbot([], label="assistant", elem_id="demobot").style(
                    height=500
                )
            with gr.Row():
                with gr.Column():
                    txt = gr.TextArea(
                        show_label=False,
                        placeholder="入力してね〜",
                        value=default_query,
                        lines=5,
                    ).style(container=False)
                with gr.Column():
                    url_txt = gr.Textbox(
                        show_label=True,
                        label="please input url ",
                        placeholder="No Memory",
                        value="https://xtech.nikkei.com/atcl/nxt/column/18/02504/062600003/",
                    ).style(container=False)
                    btn = gr.Button(value="Add Memory")

        with gr.Tab("Setting"):
            with gr.Row():
                model_dd = gr.Dropdown(
                    [
                        "gpt-3.5-turbo",
                        "gpt-3.5-turbo-0613",
                        "gpt-3.5-turbo-16k",
                        "gpt-3.5-turbo-16k-0613",
                        "gpt-4",
                        "gpt-4-0613",
                        "gpt-4-32k",
                        "gpt-4-32k-0613",
                    ],
                    value="gpt-3.5-turbo",
                    label="chat model",
                    info="you can choose the chat model.",
                )
                temperature_sl = gr.Slider(0, 200, 1, step=1, label="temperature (%)")

        txt.submit(_init_session, [status], [status]).then(
            _init, [chatbot, txt], [chatbot, txt]
        ).then(
            _chat,
            [status, chatbot, model_dd, temperature_sl],
            [status, chatbot],
        )
        btn.click(_init_memory, [status, url_txt], [status, url_txt])

    if params.do_share:
        demo.launch(
            share=True,
            auth=("user", "user123"),
            server_name="0.0.0.0",
            server_port=7860,
        )
    else:
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


# @from_config(params_file="conf/app.yml", root_key="/train")
# def config(cfg: DictConfig):
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
