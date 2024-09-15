from typing import Any, Optional, Union

# import gradio as gr
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult


class TextCallbackHandler(BaseCallbackHandler):
    def __init__(self, targets=None) -> None:
        self.targets: list = targets
        self.log_texts: list[str] = []

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any
    ) -> None:
        class_name = serialized["name"]
        if class_name and self.targets and class_name not in self.targets:
            return
        self.log_texts.append("-" * 80)
        self.log_texts.append(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        # self.log_texts.append("Finished chain.")
        pass

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        pass

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        self.log_texts.append(action.log)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if observation_prefix is not None:
            self.log_texts.append(f"\n{observation_prefix}")
        self.log_texts.append(output)
        if llm_prefix is not None:
            self.log_texts.append(f"\n{llm_prefix}")

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.log_texts.append(f"\n{error}")

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        # NOTE: prompt
        # self.log_texts.append(text)
        pass

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        self.log_texts.append(finish.log)
