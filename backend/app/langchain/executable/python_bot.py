import json
from typing import Union

from dotenv import load_dotenv
from langchain.agents import (AgentOutputParser, AgentType, Tool,
                              initialize_agent)
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AgentAction  # , OutputParserException
from langchain.schema import AgentFinish
from langchain.utilities import PythonREPL

FINAL_ANSWER_ACTION = "Final Answer:"


# NOTE: cf. /usr/local/lib/python3.10/dist-packages/langchain/agents/chat/output_parser.py
class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        try:
            action = text.split("```")[1]
            response = json.loads(action.strip())
            return AgentAction(response["action"], response["action_input"], text)

        except Exception as e:
            return AgentAction("python_repl", f"print(\"{str(e)}\")", text)
            # raise OutputParserException(f"Could not parse LLM output: {text}")


if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(temperature=0)

    python_repl = PythonREPL()
    # python_repl.run("print(1+1)")

    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. "
                    "Input should be a valid python command. "
                    "If you want to see the output of a value, "
                    "you should print it out with `print(...)`.",
        func=python_repl.run
    )

    memory = ConversationBufferMemory(return_messages=True)
    kwargs = dict(memory=memory)

    self_ask_with_search = initialize_agent(
        [repl_tool],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs=dict(
            prefix=PREFIX,
            suffix=SUFFIX.replace("Reminder to", "but remember! MUST"),
            format_instructions=FORMAT_INSTRUCTIONS,
            output_parser=CustomOutputParser(),
        ),
        **kwargs,
    )
    self_ask_with_search.run(
        "Download the langchain.com webpage and grep for all urls. "
        "Return only a sorted list of them. Be sure to use double quotes."
    )
