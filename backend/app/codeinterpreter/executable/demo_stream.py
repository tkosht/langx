import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig

from app.codeinterpreter.component.interpreter import (
    CodeInterpreter,
    CodeInterpreterResponse,
)
from app.codeinterpreter.component.llm.llm_builder import buildup_llm
from app.codeinterpreter.component.llm.schema import File
from app.codeinterpreter.component.session import (
    init_codeinterpreter,
    init_session_state,
    term_codeinterpreter,
)
from app.langchain.component.chain.stream import get_chain, get_openai_type

load_dotenv()

user_avatar = "🧐"
ai_avatar = "🐳"

st.set_page_config(
    page_title="Whale Chat",
    page_icon=ai_avatar,
    layout="wide",
    initial_sidebar_state="collapsed",
)

f"# {ai_avatar}️ Whale Chat"


# Initialize session state
init_session_state("welcome", init_value=False)
init_session_state("messages", init_value=[])
init_session_state("images", init_value=[])
init_session_state("latest_code", init_value="")
init_session_state("code_logs", init_value=[])
init_session_state("codes", init_value=[])


# ---
# Sidebar
st.sidebar.markdown(
    """
# Menu
"""
)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a cool and smart whale with the smartest AI brain. "
    "You love data science "
    "You MUST always answer in Japanese. "
)

system_prompt = st.sidebar.text_area(
    "Custom Instructions",
    _DEFAULT_SYSTEM_PROMPT,
    help="Custom instructions to provide the language model to determine style, personality, etc.",
)
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")
chain, memory = get_chain(system_prompt, temperature=0.25)

if "codeinterpreter" not in st.session_state:
    init_codeinterpreter()


def on_change_model():
    cdp: CodeInterpreter = st.session_state["codeinterpreter"]
    llm = buildup_llm(model=st.session_state.model_name)
    cdp.update_llm(llm=llm)


model_option = st.sidebar.selectbox(
    label="select the model",
    options=("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"),
    key="model_name",
    on_change=on_change_model,
)

if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    st.session_state.messages = []
    term_codeinterpreter()
    model = (
        st.session_state.model_name
        if "model_name" in st.session_state
        else "gpt-3.5-turbo"
    )
    init_codeinterpreter(model=model)


# ---
# Welcome Message holder
if not st.session_state.welcome:
    welcome_message_holder = st.chat_message("assistant", avatar=ai_avatar)
else:
    welcome_message_holder = st.empty()

# chat messages from history
ai_idx = 0
for msg in st.session_state.messages:
    streamlit_type = get_openai_type(msg)
    avatar = None
    avatar = user_avatar if streamlit_type == "user" else avatar
    avatar = ai_avatar if streamlit_type == "assistant" else avatar
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)
        if streamlit_type == "assistant":
            imgs = st.session_state.images[ai_idx]
            for img in imgs:
                st.image(image=img)
            # code = st.session_state.codes[ai_idx]
            # st.code(code, language="python", line_numbers=True)
            ai_idx += 1
    memory.chat_memory.add_message(msg)

latest_avatar_user = st.empty()
latest_avatar_ai = st.empty()

with st.container():
    with st.form(key="my_form", clear_on_submit=True):
        col1, col2 = st.columns([0.96, 0.04])
        with col1:
            message_example = "2023年までの日経平均株価を適切な前処理をした上で画像にプロットしてくれますか"
            prompt = st.text_area(
                label=f"Message: e.g.) {message_example}", key="input", value=""
            )
            uploaded_file = st.file_uploader("upload file", label_visibility="hidden")
        with col2:
            with st.empty():
                st.markdown("<br>" * 3, unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="🫧")


def parse_response(response: CodeInterpreterResponse):
    text = ""
    img = None
    imgs = []

    try:
        text: str = response.content

        imgs = []
        for fl in response.files:
            img = fl.get_image()
            if img.mode not in ("RGB", "L"):  # L is for greyscale images
                img = img.convert("RGB")
            imgs.append(img)
    except Exception as e:
        print(e)
        if not text:
            text = str(e)

    return text, imgs


run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

if submit_button and prompt:
    with latest_avatar_user:
        with st.chat_message("user", avatar=user_avatar):
            st.write(f"{prompt}")

    with latest_avatar_ai:
        with st.chat_message("assistant", avatar=ai_avatar):
            _msg_area_ai = st.empty()
            _image_area_ai = st.empty()
            _status_area = st.empty()
            _code_area = st.empty()

        st.session_state.latest_code = ""

        def log_handler(text: str, is_code=False):
            if is_code:
                with _code_area:
                    st.code(text, language="python", line_numbers=True)
                    st.session_state.latest_code = text
                    st.session_state.code_logs.append(text)
            else:
                with _status_area:
                    st.markdown(text)

        files = []
        if uploaded_file:
            fl = File(name=uploaded_file.name, content=uploaded_file.getvalue())
            files.append(fl)

        with _msg_area_ai:
            log_handler("入力プロンプトを英語にしています・・・")
            with st.spinner("ちょっとまっててー"):
                msg = f"""以下を英語にしてください。翻訳した結果の英語のみを返してください。
                ```
                {prompt}
                ```"""
                user_msg = "".join(
                    [
                        chunk.content
                        for chunk in chain.stream(
                            {"input": msg}, config=runnable_config
                        )
                    ]
                )
                print("-" * 50)
                print(f"{user_msg=}")

                cdp: CodeInterpreter = st.session_state["codeinterpreter"]
                cdp.update_log_handler(log_handler=log_handler)

                print("-" * 50)
                print("llm:", cdp.llm.model_name)

                cdp.reconnect()
                print("CodeInterpreter: reconnected")

                log_handler(f"処理中です・・・ {cdp.llm.model_name}: {user_msg}")
                response = cdp.generate_response_sync(user_msg=user_msg, files=files)
                _text, imgs = parse_response(response)

            # NOTE: into japanese
            msg = f"""以下を日本語にしてください。翻訳した結果の日本語のみを返してください。エラーメッセージの場合はへそのまま返してください。
            ```
            {_text}
            ```"""

            log_handler("完了しました")

            text: str = ""
            st.markdown(text + "▌")

            # streaming output
            for chunk in chain.stream({"input": msg}, config=runnable_config):
                text += chunk.content
                st.markdown(text + "▌")
            st.markdown(text)

        with _image_area_ai:
            for img in imgs:
                st.image(image=img)

        log_handler("")

    memory.save_context({"input": prompt}, {"output": text})
    st.session_state.messages = memory.buffer
    st.session_state.images.append(imgs)
    st.session_state.codes.append(st.session_state.latest_code)

# make welcome message
if not st.session_state.welcome:
    with welcome_message_holder:
        with st.empty():
            msg = "You have to make a humble welcome message to the user to support, politely and friendly in japanese."
            welcome_message = ""
            for chunk in chain.stream({"input": msg}):
                welcome_message += chunk.content
                st.markdown(welcome_message)
    st.session_state.welcome = True
