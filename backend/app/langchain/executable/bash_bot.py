from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.chat_models import ChatOpenAI
from langchain.tools import ShellTool

if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(temperature=0)

    shell_tool = ShellTool()
    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")
    self_ask_with_search = initialize_agent(
        [shell_tool],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs=dict(
            prefix=PREFIX,
            suffix=SUFFIX.replace("Reminder to", "but remember! MUST"),
            format_instructions=FORMAT_INSTRUCTIONS,
        ),
    )
    self_ask_with_search.run(
        "Download the langchain.com webpage and grep for all urls. "
        "Return only a sorted list of them. Be sure to use double quotes."
    )
