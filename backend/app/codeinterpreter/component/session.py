import streamlit as st

from app.codeinterpreter.component.interpreter import CodeInterpreter
from app.codeinterpreter.component.llm.llm_builder import buildup_llm


def init_session_state(key: str, init_value):
    if key not in st.session_state:
        st.session_state[key] = init_value
    return st.session_state[key]


def init_codeinterpreter(model: str = "gpt-3.5-turbo"):
    llm = buildup_llm(model=model)
    st.session_state["codeinterpreter"] = cdp = CodeInterpreter(
        llm=llm, local=True, verbose=True
    )
    cdp.start()


def term_codeinterpreter():
    cdp: CodeInterpreter = st.session_state["codeinterpreter"]
    cdp.stop()
