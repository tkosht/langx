import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.tracers.run_collector import \
    RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig

from app.langchain.component.chain.stream import get_chain, get_openai_type

load_dotenv()


user_avatar = "🙂"
ai_avatar = "🐳"

st.set_page_config(
    page_title="Chat LangSmith",
    page_icon=ai_avatar,
    layout="wide",
    initial_sidebar_state="collapsed",
)

f"# {ai_avatar}️ Whale Chat"

# Initialize State
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# setup slidebar
st.sidebar.markdown(
    """
# Menu
"""
)
if st.sidebar.button("Clear message history"):
    print("Clearing message history")
    st.session_state.messages = []


# setup prompt
_DEFAULT_SYSTEM_PROMPT = (
    "You are a cool and smart whale with the smartest AI brain. "
    "You love programming, coding, mathematics, Japanese, and friendship!"
)

system_prompt = st.sidebar.text_area(
    "Custom Instructions",
    _DEFAULT_SYSTEM_PROMPT,
    help="Custom instructions to provide the language model to determine style, personality, etc.",
)
system_prompt = system_prompt.strip().replace("{", "{{").replace("}", "}}")
chain, memory = get_chain(system_prompt)


# handler
run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

# chat messages from history
for msg in st.session_state.messages:
    streamlit_type = get_openai_type(msg)
    avatar = None
    avatar = user_avatar if streamlit_type == "user" else avatar
    avatar = ai_avatar if streamlit_type == "assistant" else avatar
    with st.chat_message(streamlit_type, avatar=avatar):
        st.markdown(msg.content)
    memory.chat_memory.add_message(msg)

latest_avatar_user = st.empty()
latest_avatar_ai = st.empty()

container = st.container()
with container:
    with st.form(key="my_form", clear_on_submit=True):
        col1, col2 = st.columns([0.96, 0.04])
        with col1:
            prompt = st.text_area(label="Message: ", key="input")
        with col2:
            with st.empty():
                st.markdown("<br>" * 3, unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="🫧")

if submit_button and prompt:
    with latest_avatar_user:
        with st.chat_message("user", avatar=user_avatar):
            st.write(f"{prompt}")

    # chat streaming
    with latest_avatar_ai:
        with st.chat_message("assistant", avatar=ai_avatar):
            _msg_area_ai = st.empty()

        with _msg_area_ai:
            full_response: str = ""
            for chunk in chain.stream({"input": prompt}, config=runnable_config):
                full_response += chunk.content
                st.markdown(full_response + "▌")
            st.markdown(full_response)

    memory.save_context({"input": prompt}, {"output": full_response})
    st.session_state.messages = memory.buffer
