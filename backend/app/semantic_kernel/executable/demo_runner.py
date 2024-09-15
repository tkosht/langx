from inspect import signature

import gradio as gr
import typer
from omegaconf import DictConfig

from app.semantic_kernel.component.planner import CustomPlanner
from app.semantic_kernel.component.runner import SimpleRunner


def _init(
    history: list[tuple[str, str]], text: str
) -> tuple[list[tuple[str, str]], str]:
    history = history + [(text, None)]
    return history, ""


async def _init_session(status: dict) -> dict:
    import uuid

    if "session_id" in status:
        return status

    status["session_id"] = str(uuid.uuid4())
    skill_dir = "./app/semantic_kernel/component/skills/"
    runner = SimpleRunner(planner=CustomPlanner(), skill_dir=skill_dir)
    await runner.astart()
    status["runner"] = runner

    return status


async def _run(
    status: dict,
    history: list[tuple[str, str]],
    model_name: str,
    temperature_percent: int,  # in [0, 200]
    max_tokens: int = 1024,
    n_window: int = 3,
) -> list[tuple[str, str]]:
    input_query: str = history[-1][0]
    temperature: float = temperature_percent / 100

    runner: SimpleRunner = status["runner"]
    runner.set_planner(
        planner=CustomPlanner(temperature=temperature, max_tokens=max_tokens)
    )
    runner.update_kernel_model(model_name=model_name)

    # NOTE: たぶん、memory を使わないほうが賢い動きになるかも(文脈の必要なところをLLMに任せられるため)
    context = "---\n\n".join(
        [f"\norder: {q}answer: {a}" for q, a in history[-1 - n_window : -1]]
    )
    context = context.replace("\n", "\n    ")

    # make user query
    user_query = f"""`これまでのユーザの依頼と回答(文脈)` を可能な範囲で踏まえて、`今回のユーザの依頼` に応えてください。

    # 今回のユーザの依頼
    <<<
    {input_query}
    >>>

    # これまでのユーザの依頼と回答(文脈)
    <<<
    {context if context else "なし"}
    >>>
"""

    response: str = await runner.do_run(user_query=user_query, n_retries=3)
    img = None

    try:
        import json

        resdic = json.loads(response)
        if "text" in resdic:
            response: str = resdic["text"]
        if "images" in resdic:
            import base64
            from io import BytesIO

            from PIL import Image

            for img64 in resdic["images"]:
                img_io = BytesIO(base64.b64decode(img64.encode()))
                img = Image.open(img_io)
                if img.mode not in ("RGB", "L"):  # L is for greyscale images
                    img = img.convert("RGB")
                break

    except Exception as e:
        print(e)
        pass

    # keep memory
    history[-1][1] = response

    return status, history, img


def _main(params: DictConfig):
    # default_query = "今日の川崎の天気を教えてくれませんか？"
    # default_query = "今日の大規模言語モデル(LLM)に関するニュースを調べて情報源を含めて正確にわかりやすくまとめてくれますか？"
    default_query = "Plot the bitcoin chart of 2023 YTD"

    with gr.Blocks() as demo:
        status = gr.State({})

        with gr.Tab("Conversation"):
            with gr.Row():
                height: int = 500
                with gr.Column():
                    chatbot = gr.Chatbot(
                        [], label="assistant", elem_id="demobot", height=height
                    )
                with gr.Column():
                    img = gr.outputs.Image(type="pil").style(height=height)

            with gr.Row():
                with gr.Column():
                    txt = gr.TextArea(
                        show_label=False,
                        placeholder="入力してね〜",
                        value=default_query,
                        lines=5,
                        container=False,
                    )

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

        txt.submit(_init, [chatbot, txt], [chatbot, txt]).then(
            _init_session, [status], [status]
        ).then(
            _run, [status, chatbot, model_dd, temperature_sl], [status, chatbot, img]
        )

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
