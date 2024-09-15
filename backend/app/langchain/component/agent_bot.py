import base64
import time
from io import BytesIO

import PIL.Image
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory  # ChatMessageHistory
from langchain.schema import AgentAction
from langchain.schema.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.base.component.ulid import build_ulid
from app.langchain.component.agent.agent_builder import build_agent
from app.langchain.component.agent.agent_executor import (  # AgentExecutor,
    CustomAgentExecutor,
)
from app.langchain.component.callbacks.simple import TextCallbackHandler


def _update_text(log_text: str, cb: TextCallbackHandler):
    return "\n".join(cb.log_texts)


def encode_image_from_pil(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class SimpleBot(object):
    def __init__(self, model_name: str = "gpt-4-turbo") -> None:
        load_dotenv()
        self.session_id = build_ulid(prefix="SSN")
        self.intermediate_steps: list[tuple[AgentAction, str]] = []
        self._callback = TextCallbackHandler(targets=["CustomAgentExecutor"])
        self.llm = ChatOpenAI(model=model_name, max_tokens=2048)
        self.memory = ConversationBufferMemory(return_messages=True)
        # self.memory = ChatMessageHistory(return_messages=True)

    def chat_with_image(self, query: str, img: PIL.Image = None) -> str:
        base64_image = encode_image_from_pil(img)
        msg = self.llm.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ]
                )
            ]
        )
        return msg.content

    def gr_chat(
        self,
        history: list[str],
        model_name: str,
        temperature_percent: int,
        max_iterations: int,
        img: PIL.Image = None,
    ):
        query = history[-1][0]
        is_initial_qa = len(history) == 1
        n_context: int = 10

        context = "\n".join([f"Q: [{q}]\nA: [{a}]" for q, a in history[:-1][-n_context:]])
        if context:
            _query = f"""'これまでのQA' (特に直前のA: の結果と直前のA: に関連するQA)を踏まえた上で、'今回の依頼' を5W1H に即して具体的かつ正確な質問や依頼になるように書き直してくれませんか？
特に、番号や記号、単語は直前の A: に関連するものとして扱ってください。

今回の依頼: <
{query}
>

これまでのQA: <
{context}
>

出力フォーマット: <
主なテーマ: <<これまでのQAを踏まえて修正した今回の依頼内容の主なテーマや大前提となる事柄を単語で記載します>>
今回の依頼: <<これまでのQAを踏まえて修正した今回の依頼内容を記載します>>
>
"""
            simplified = self.llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": _query},
                        ]
                    )
                ]
            )
            query = simplified.content.replace("出力フォーマット: ", "")
            print(f"{simplified.content=}")
            print(f"modified {query=}")

        query = f"""
今回の依頼: <
{query}
>

これまでのQA: <
{context}
>

制約: <
日本語で回答します
>
"""
        answer = self.run(
            query=query,
            model_name=model_name,
            temperature_percent=temperature_percent,
            max_iterations=max_iterations,
            img=img,
            is_initial_qa=is_initial_qa,
        )
        assert answer
        history[-1][1] = answer  # may be changed url to href

        return history

    def run_executor(
        self,
        query_: str,
        model_name: str,
        temperature_percent: int,
        max_iterations: int,
        max_retries: int = 7,
    ) -> str:
        agent_executor: CustomAgentExecutor = build_agent(
            model_name,
            temperature=temperature_percent / 100,
            max_iterations=max_iterations,
            memory=self.memory,
        )

        query_org = query_
        intermediate_steps = []

        for _ in range(max_retries):
            try:
                params = dict(
                    input=query_,
                    callbacks=[self._callback],
                )
                if hasattr(agent_executor, "intermediate_steps"):
                    params.update(dict(intermediate_steps=intermediate_steps))

                answer = agent_executor.run(**params)
                break
            except Exception as e:
                print("-" * 80)
                print(f"# {e.__repr__()} / in run()")
                print("-" * 80)
                answer = f"Error Occured: '{e}' / couldn't be fixed."
                query_ = f"Observation: \nERROR: {str(e)}\n\nwith fix this Error in other way, "
                f"\n\nHUMAN: {query_org}\nThougt:"
                if not hasattr(agent_executor, "intermediate_steps"):
                    continue
                if "This model's maximum context length is" in str(e):
                    n_context = 1
                    if len(agent_executor.intermediate_steps) <= n_context:
                        # NOTE: already 'n_context' intermediate_steps, but context length error. so, restart newly
                        query_ = query_org
                        agent_executor.intermediate_steps = []
                        # agent_executor.memory.clear()
                    else:
                        agent_executor.intermediate_steps = agent_executor.intermediate_steps[-n_context:]
                        # agent_executor.memory.chat_memory.messages = agent_executor.memory.chat_memory.messages[:1]
                else:
                    # NOTE: retry newly
                    query_ = query_org
                    agent_executor.intermediate_steps = []

            finally:
                print("<" * 80)
                print("Next:")
                print(f"{query_=}")
                print("-" * 20)
                if hasattr(agent_executor, "intermediate_steps"):
                    intermediate_steps = agent_executor.intermediate_steps
                    print(f"{len(agent_executor.intermediate_steps)=}")
                print(">" * 80)
            print("waiting retrying ... (a little sleeping)")
            time.sleep(1)

        return answer

    def run(
        self,
        query: str,
        model_name: str,
        temperature_percent: int,
        max_iterations: int,
        img: PIL.Image = None,
        is_initial_qa: bool = False,
    ) -> str:
        self.model_name = model_name
        self.temperature_percent = temperature_percent
        self.max_iterations = max_iterations

        answer = "(Nothing)"
        try:
            if is_initial_qa and img is not None:
                answer = self.chat_with_image(query, img)
                return answer

            answer = self.run_executor(query, model_name, temperature_percent, max_iterations)
            return answer

        finally:
            print("-" * 120)
            print(f"{answer=}")
            print("-" * 120)

    def gr_clear_context(self, context: str):
        self.intermediate_steps.clear()
        return "\n".join(self.intermediate_steps)

    def gr_update_text(self, log_text: str):
        log = _update_text(log_text, self._callback)
        return log

    def gr_clear_text(self):
        self._callback.log_texts.clear()
        log = _update_text("", self._callback)
        return log
