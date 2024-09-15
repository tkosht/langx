from langchain.tools import ShellTool


class CustomShellTool(ShellTool):
    def __init__(self) -> None:
        super().__init__()
        self.description += f"args {self.args}".replace("{", "{{").replace("}", "}}")
