import sys
from io import StringIO

from langchain.utilities import PythonREPL


class CustomPythonREPL(PythonREPL):
    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # NOTE: support multiple commands
        command_list: list[str] = [command]
        if isinstance(command, list):
            command_list: list[str] = command

        output: str = ""
        try:
            for cmd in command_list:
                exec(cmd, self.globals, self.locals)
                sys.stdout = old_stdout
                o = mystdout.getvalue() + "\n"
                output += o
        except Exception as e:
            sys.stdout = old_stdout
            output = f"{type(e).__name__} (python_repl): " + str(e.args)

        n_cutoff = 5
        logs = output.split("\n")
        if len(logs) > n_cutoff*2 + 1:
            # NOTE: 長すぎたら、最初と最後だけに省略する
            _logs = logs[:n_cutoff] + ["...\n"] + logs[-n_cutoff:]
            output = "\n".join(_logs)
        output = output.strip()
        return output
