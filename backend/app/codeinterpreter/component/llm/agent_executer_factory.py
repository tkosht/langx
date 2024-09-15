from typing import Callable

from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool, StructuredTool

from app.codeinterpreter.component.llm.agents import OpenAIFunctionsAgent
from app.codeinterpreter.component.llm.prompts import code_interpreter_system_message
from app.codeinterpreter.component.llm.schema import CodeInput


def create_tools(
    run_handler: Callable, additional_tools: list[BaseTool]
) -> list[BaseTool]:
    return additional_tools + [
        StructuredTool(
            name="python",
            description="Input a string of code to a ipython interpreter. "
            "Write the entire code in a single string. This string can "
            "be really long, so you can use the `;` character to split lines. "
            "Variables are preserved between runs. ",
            func=run_handler,
            # coroutine=self._arun_handler,
            args_schema=CodeInput,
        ),
    ]


def create_agent_executor(
    llm: BaseLanguageModel,
    tools: list[BaseTool],
    max_iterations: int = 10,
    memory: ConversationBufferMemory = None,
    callback_manager: BaseCallbackManager = None,
    verbose: bool = False,
) -> AgentExecutor:
    # NOTE: no specfy the memory, then create a memory
    memory = memory or ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=ChatMessageHistory(),
    )
    return AgentExecutor.from_agent_and_tools(
        agent=_create_agent(llm, tools),
        max_iterations=max_iterations,
        tools=tools,
        memory=memory,
        callback_manager=callback_manager,
        verbose=verbose,
    )


def _create_agent(
    llm: BaseLanguageModel, tools: list[BaseTool]
) -> BaseSingleActionAgent:
    # from langchain.agents import AgentOutputParser

    return OpenAIFunctionsAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=code_interpreter_system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
        # output_parser=AgentOutputParser(),
    )
