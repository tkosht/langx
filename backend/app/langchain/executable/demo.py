from inspect import signature

import gradio as gr
import typer
from gradio import Brush
from omegaconf import DictConfig

from app.langchain.component.agent_bot import SimpleBot


def _init(history: list[tuple[str, str]], text: str):
    history = history + [(text, None)]
    return history, "", ""


def _main(params: DictConfig):
    _prompt_example = (
        "Download the langchain.com webpage and grep for all urls. "
        "Return only a sorted list of them. Be sure to use double quotes."
    )
    # 1-2. 例えば "KaggleチュートリアルTitanicで上位2%以内に入るノウハウ" などのWebサイトが役に立ちそうです
    _prompt_default = """
# 依頼内容

titanic dataset に対して、scikit-learn の LightGBM を使ってクラス分類し最高精度を達成する Python コードを作成してください

# 要件

1. 手法について十分な調査を行い、精度指標値が83%以上になるような情報を収集し整理します
    1. 精度指標値が83%以上になるようにWeb上からベストプラクティスを収集してください
        1-1. 特に、精度を上げるためのパラメータチューニングや特徴量エンジニアリングなどを中心に具体的に調査します
            1-1-1. 具体的な特徴量エンジニアリングの手法を記述してください
            1-1-2. 具体的な手法やツール名、パラメータなどを記述してください
    2. 調査した内容のポイントを整理し手順を記載してください
    3. その他、必要な情報収集を行い、集めた情報を整理して手順に追記していきます
2. 作成した手順を元に実際に、シミュレーションをします
    2-1. titanic dataset をダウンロードして 'data/titanic.csv' として上書き保存してください
    2-2. scikit-learn の LightGBM を使ってクラス分類し精度指標値を出力する Python コードを作成してください
    2-3. 作成した Python コードを 'result/titanic.py' として必ず保存します
    2-4. `2.` で作成したPythonコードを実際に実行結果をシミュレーションし精度指標値を確認してください
    2-5. 精度指標値は、accuracy score で評価してください
3. シミュレーションした結果を元に、精度を上げるためのパラメータチューニングや特徴量エンジニアリングなどで精度指標値を改善し目標精度値を達成するまで上記1.～2. を繰り返します
4. 十分な精度が達成できたら、その Python コードを 'result/titanic_best.py' というローカルファイルに上書き保存してください

# 制約

1. 利用可能なツールは、`python_repl` ツール または `bash/terminal` ツールのみです
2. エラーが発生したら、要因分析を行い修正・改善を繰り返してください
    1-1. 必要に応じて、Web 上でエラーメッセージを検索し、解決策を探してください

"""  # noqa
    _prompt_default = ""

    bot = SimpleBot()

    with gr.Blocks() as demo:
        with gr.Tab("Conversation"):
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot([], label="assistant", elem_id="demobot", height=900)

                with gr.Column():
                    image_area = gr.ImageEditor(
                        value=None,
                        type="pil",
                        height=708,  # width=400,
                        label="image",
                        brush=Brush(default_color="red", default_size=10),
                    )
                    image_hidden = gr.Image(visible=False, type="pil")

                    def on_change(im: dict):
                        assert "composite" in im, "composite not found in im"
                        return im["composite"]

                    image_area.change(on_change, inputs=[image_area], outputs=[image_hidden])

                    txt = gr.TextArea(
                        show_label=False,
                        placeholder=_prompt_example,
                        value=_prompt_default,
                        container=False,
                        lines=7,
                    )
                # with gr.Column():
                #     log_area = gr.TextArea(
                #         lines=21,
                #         max_lines=21,
                #         show_label=False,
                #         label="log",
                #         placeholder="",
                #         value="",
                #         container=False,
                #     )
                #     with gr.Row():
                #         btn = gr.Button(value="update agent log")
                #         btn.click(
                #             bot.gr_update_text, inputs=[log_area], outputs=[log_area]
                #         )

        with gr.Tab("Setting"):
            with gr.Row():
                model_dd = gr.Dropdown(
                    [
                        "gpt-4-turbo",
                        "gpt-4-turbo-2024-04-09",
                        "gpt-4",
                        "gpt-4-0314",
                        "gpt-4-32k-0314",
                        "gpt-4-32k",
                        "gpt-3.5-turbo-16k",
                        "gpt-3.5-turbo",
                    ],
                    # value="gpt-3.5-turbo-16k",
                    value="gpt-4-turbo",
                    label="chat model",
                    info="you can choose the chat model.",
                )
                temperature_sl = gr.Slider(0, 100, 10, step=1, label="temperature (%)")
                max_iterations_sl = gr.Slider(0, 50, 5, step=1, label="max_iterations")

        txt.submit(_init, [chatbot, txt], [chatbot, txt]).then(
            bot.gr_clear_text,
            [],  # [log_area]
        ).then(
            bot.gr_chat,
            [
                chatbot,
                model_dd,
                temperature_sl,
                max_iterations_sl,
                image_hidden,
            ],
            [chatbot],
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
