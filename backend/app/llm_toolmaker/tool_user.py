import os
import random
import re
import time

import openai
from dotenv import load_dotenv

from app.base.component.logger import Logger

g_logger = Logger(logger_name="app")
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# cf. https://github.com/ctlllll/LLM-ToolMaker.git


def pickup_func(wrapper: str):
    func = re.findall(r"```python\n(.*?)\n```", wrapper, re.DOTALL)[0]
    return func


class LlmToolUser(object):
    def __init__(
        self,
        wrapper: str,
        task_name: str = "example_task",
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        n_retry: int = 3,
    ) -> None:
        self.wrapper: str = wrapper
        self.task_name: str = task_name
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.n_retry: int = n_retry

        assert self.model_name[: len("gpt")] == "gpt"

    def _params(self, messages: list[dict]):
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        return params

    def _llm(self, **params):
        response = openai.ChatCompletion.create(**params)["choices"][0]["message"][
            "content"
        ]
        return response

    def generate(self, prompt):
        params = self._params(messages=[{"role": "user", "content": prompt}])
        for n_retry in range(self.n_retry):
            try:
                return self._llm(**params)
            except Exception as e:
                if "Rate limit" in " ".join(e.args):
                    sleep_seconds = 15 + 2**n_retry + random.random()
                    errmsg = re.sub(
                        r"org-\w+", "org-" + ("x" * 24), f"{e}"
                    )  # masking for secure
                    g_logger.warning(f"{errmsg} ... try to retry [{sleep_seconds=}]")
                    time.sleep(sleep_seconds)
                else:
                    g_logger.warning(f"{e} ... try to retry")
        raise Exception("Failed to generate")

    def make_answer_from_sample(self, task: str, sample: dict):
        prompt = self.wrapper + "\n\nQuestion: " + sample["question"]
        ans = self.make_answer(prompt=prompt)

        return ans

    def make_answer(self, prompt: str):
        caller = self.generate(prompt)
        func_call = pickup_func(caller)
        func_def = pickup_func(self.wrapper)

        exec_code = func_def + "\n" + func_call
        _ = exec(exec_code, globals())  # output: printed texts
        answer_variable = re.findall(r"(ans.*?) =", func_call, re.DOTALL)[-1]
        ans = globals()[answer_variable]

        return ans
