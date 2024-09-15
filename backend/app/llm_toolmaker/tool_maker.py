import json
import os
import pathlib
import re
from dataclasses import dataclass

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig
from typing_extensions import Self

from app.base.component.logger import Logger
from app.llm_toolmaker.prompt import (
    tool_maker_prompt,
    tool_test_prompt,
    tool_wrapper_prompt,
)

g_logger = Logger(logger_name="app")
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


@dataclass
class LlmTool(object):
    toolcode: str
    testcode: str
    wrapper: str


# cf. https://github.com/ctlllll/LLM-ToolMaker.git
class LlmToolMaker(object):
    def __init__(
        self,
        task_name: str = "example_task",
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        n_retry: int = 3,
    ) -> None:
        self.task_name: str = task_name
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.n_retry: int = n_retry

        self._llm_messages: list = []

        assert self.model_name[: len("gpt")] == "gpt"

    def _params(self):
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": self._llm_messages,
        }
        return params

    def _llm(self, **params):
        response = openai.ChatCompletion.create(**params)["choices"][0]["message"][
            "content"
        ]
        return response

    def buildup_tool(
        self,
        toolmaker_prompt: str = tool_maker_prompt,
        tooltest_prompt: str = tool_test_prompt,
        toolwrapper_prompt: str = tool_wrapper_prompt,
    ) -> LlmTool:
        toolcode = self.make_toolcode(toolmaker_prompt)
        g_logger.info(f"{toolcode=}")

        testcode = self.make_testcode(
            toolcode=toolcode,
            tooltest_prompt=tooltest_prompt,
        )
        g_logger.info(f"{testcode=}")

        wrapper = self.make_wrappercode(toolwrapper_prompt=toolwrapper_prompt)
        self.dump()

        tool = LlmTool(toolcode=toolcode, testcode=testcode, wrapper=wrapper)
        return tool

    def buildup_tool_from_datasets(
        self,
        trainset: list,  # using for making tool code
        validset: list,  # using for making test code
        n_train_samples: int = None,
        n_valid_samples: int = None,
        toolmaker_prompt: str = tool_maker_prompt,
        tooltest_prompt: str = tool_test_prompt,
        toolwrapper_prompt: str = tool_wrapper_prompt,
    ) -> LlmTool:
        _toolmaker_prompt = (
            "\n\n".join(
                [
                    f"Question: {sample['question']}\nAnswer: {sample['answer']}"
                    for sample in trainset[:n_train_samples]
                ]
            )
            + "\n\n"
            + toolmaker_prompt
        )
        _tooltest_prompt = (
            "\n\n".join(
                [
                    f"Question: {sample['question']}\nAnswer: {sample['answer']}"
                    for sample in validset[:n_valid_samples]
                ]
            )
            + "\n\n"
            + tooltest_prompt
        )
        return self.buildup_tool(
            toolmaker_prompt=_toolmaker_prompt,
            tooltest_prompt=_tooltest_prompt,
            toolwrapper_prompt=toolwrapper_prompt,
        )

    def make_toolcode(self, toolmaker_prompt: str, max_tokens: int = 2048) -> str:
        self._llm_messages = [{"role": "user", "content": toolmaker_prompt}]
        params = self._params()  # using latest self._llm_messages
        for _ in range(self.n_retry):
            try:
                response: str = self._llm(**params)
                self._llm_messages.append({"role": "assistant", "content": response})
                toolcode: str = "\n\n".join(
                    re.findall(r"```python\n(.*?)```", response, re.DOTALL)
                )
                _ = exec(toolcode, globals())  # output: printed texts
                break
            except Exception as e:
                g_logger.warning("Failed to generate tool", e)
                self._llm_messages.append(
                    {
                        "role": "user",
                        "content": f"Failed to execute the function due to the error: {type(e).__name__} {e}. "
                        "Please fix it and try again.",
                    }
                )
        return toolcode

    def make_testcode(self, toolcode: str, tooltest_prompt: str) -> str:
        self._llm_messages.append({"role": "user", "content": tooltest_prompt})
        params = self._params()  # using latest self._llm_messages

        success = False
        for _ in range(self.n_retry):
            try:
                response = self._llm(**params)
                self._llm_messages.append({"role": "assistant", "content": response})
                testcode = "\n\n".join(
                    re.findall(r"```python\n(.*?)```", response, re.DOTALL)
                )
                unittest = toolcode + "\n" + testcode
                _ = exec(unittest, globals())  # output: printed texts
                success = True
                break
            except Exception as e:
                g_logger.warning("Failed to the simple tooltest", e)
                self._llm_messages.append(
                    {
                        "role": "user",
                        "content": f"Failed to verify the function due to the error: {type(e).__name__} {e}. "
                        "Please fix it and try again.",
                    }
                )
            if not success:
                raise Exception(
                    f"Failed to make tooltest code for the toolcode [{toolcode}], "
                    f"the last testcode: {testcode}"
                )
        return testcode

    def make_wrappercode(self, toolwrapper_prompt: str) -> None:
        self._llm_messages.append({"role": "user", "content": toolwrapper_prompt})
        params = self._params()  # using latest self._llm_messages
        try:
            wrapper = self._llm(**params)
            self._llm_messages.append({"role": "assistant", "content": wrapper})
            g_logger.info("Wrapper:", wrapper)
        except Exception as e:
            g_logger.error("Failed to generate wrapper", e)
            raise e
        return wrapper

    def dump(self, tooldir="./llm_tools") -> Self:
        pathlib.Path(tooldir).mkdir(parents=True, exist_ok=True)
        json_file = f"{tooldir}/{self.task_name}.json"
        with open(json_file, "w") as f:
            json.dump(self._llm_messages, f)
        return self


def _main(params: DictConfig):
    from app.llm_toolmaker.bbh import get_task

    g_logger.info(f"{params.task_name=}")
    trainset, validset, testset = get_task(task=params.task_name)

    ltm = LlmToolMaker(task_name=params.task_name)
    tool = ltm.buildup_tool_from_datasets(trainset, validset)
    print(f"{tool=}")


# @from_config(params_file="conf/app.yml", root_key="/train")
# def config(cfg: DictConfig):
def config():
    cfg = DictConfig(dict(task_name="word_sorting"))
    return cfg


def main(
    task_name: str = None,
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
    from inspect import signature

    import typer

    typer.run(main)
