import asyncio
import threading

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header

from app.semantic_kernel.component.planner import CustomPlanner
from app.semantic_kernel.component.runner import SimpleRunner
from app.semantic_kernel.component.semantic_memory import (
    SemanticMemory,
)  # SemanticMemorySummarizer,

load_dotenv()

img = Image.open("data/logo.jpeg")
img = img.resize((32, 32))
st.set_page_config(page_title="Runner Chat", page_icon=img)


with st.sidebar:
    st.title("💬Runner Chat")
    query_example = """今日の大規模言語モデル(LLM)に関するニュースを調べて情報源を含めて正確にわかりやすくまとめてくれますか？"""
    label_example = """example:"""
    model_name = st.selectbox(
        "select the model",
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
    )
    st.text_input(label_example, value=f"{query_example}", disabled=True)
    add_vertical_space(5)
    st.write("by [tkosht](https://github.com/tkosht)")


if "bot_chat" not in st.session_state:
    st.session_state["bot_chat"] = []

if "user_chat" not in st.session_state:
    st.session_state["user_chat"] = []


# Layout of input/response containers
response_container = st.container()
colored_header(label="", description="", color_name="blue-30")
input_container = st.container()


def submit():
    asyncio.run(_submit())


async def _submit():
    if st.session_state.is_running:
        # Do Nothing
        return

    try:
        # user
        st.session_state.is_running = True
        user_input = st.session_state.user_input
        st.session_state.user_chat.append(user_input)

        user_message = user_input
        # st.session_state.user_input = ""

        # bot
        response = await generate_response(user_message)
        st.session_state.bot_chat.append(response)
    finally:
        st.session_state.is_running = False


# User input
def get_text(input_text):
    label = """メッセージを入力してね✨"""
    user_input = input_text.text_input(
        label, value="", key="user_input"  # , on_change=submit
    )
    return user_input


with input_container:
    input_text = st.empty()
    user_input = get_text(input_text)


g_lock_img = threading.Lock()

if "image_chache" not in st.session_state:
    st.session_state["image_chache"] = {}


if "is_running" not in st.session_state:
    st.session_state["is_running"] = False


if "memory" not in st.session_state:
    st.session_state["memory"] = SemanticMemory()
    st.session_state["history"]: list[str] = []
    skill_dir = "./app/semantic_kernel/component/skills/"
    st.session_state["runner"] = SimpleRunner(
        planner=CustomPlanner(), skill_dir=skill_dir, model_name=model_name
    )


# Response output
async def generate_response(prompt: str):
    if prompt.lower()[: len("hello")] == "hello":
        return "hello"

    runner: SimpleRunner = st.session_state.runner
    runner.setup_kernel(model_name=model_name)  # update model
    history: list[str] = st.session_state.history
    memory: SemanticMemory = st.session_state.memory

    with st.spinner(text="Now loading memory..."):
        context = ""
        if history:
            if len(history) < 5:
                context = "\n\n".join(history)
            else:
                search_results = await memory.search(query=prompt)
                for r in search_results:
                    context += f"\n{r.text}\n"

    user_query = f"""`これまでのユーザの依頼と回答` を可能な範囲で踏まえて、`今回のユーザの依頼` に応えてください。

# 今回のユーザの依頼
<<<
{prompt}
>>>

# これまでのユーザの依頼と回答
<<<
{context if context else "なし"}
>>>
"""

    with st.spinner(text="Now running..."):
        print(user_query)
        response = await runner.do_run(user_query=user_query)

    memory_text = f"""
order: {prompt}
response: {str(response)}
"""
    await memory.append(text=memory_text)

    history.append(memory_text)
    return response


class ChatWriter(object):
    async def load_logo(self, logo_file: str):
        from PIL import Image

        if logo_file in st.session_state.image_chache:
            return st.session_state.image_chache[logo_file]

        img = Image.open(logo_file)
        img = img.resize((32, 32))

        with g_lock_img:
            st.session_state.image_chache.update({logo_file: img})

        return img

    async def chat_message(self, message: str, logo_file: str):
        logo = await self.load_logo(logo_file=logo_file)
        col_rates = [0.05, 0.95]
        col1, col2 = st.columns(col_rates)
        with col1:
            st.image(logo)
        with col2:
            st.write(message)


chatter = ChatWriter()


async def do_chat():
    logo_file_user = "data/logo.jpeg"
    logo_file_bot = "data/logo_bot.jpeg"

    await chatter.load_logo(logo_file=logo_file_user)
    await chatter.load_logo(logo_file=logo_file_bot)

    response_container.empty()
    with response_container:
        for i in range(len(st.session_state["bot_chat"])):
            # user
            user_message = st.session_state["user_chat"][i]
            await chatter.chat_message(user_message, logo_file=logo_file_user)

            # bot
            bot_message = st.session_state["bot_chat"][i]
            await chatter.chat_message(bot_message, logo_file=logo_file_bot)

    if user_input:
        await _submit()


with st.spinner(text="Now loading..."):
    asyncio.run(do_chat())
