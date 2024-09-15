"""
このコードは `Dominic Bäumer` 氏 のプロジェクト(https://github.com/shroominic/codeinterpreter-api.git)を参考に作成しました。

オリジナルのライセンス:
- MIT License
"""
import base64
import re
import traceback
from io import BytesIO
from typing import Callable, Optional
from uuid import UUID, uuid4

from codeboxapi.schema import CodeBoxOutput, CodeBoxStatus
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool
from typing_extensions import Self

from app.codebox.localbox import CustomLocalBox
from app.codeinterpreter.component.llm.agent_executer_factory import (
    create_agent_executor,
    create_tools,
)
from app.codeinterpreter.component.llm.chains import (
    get_file_modifications,
    remove_download_link,
)
from app.codeinterpreter.component.llm.schema import (
    CodeInterpreterResponse,
    File,
    UserRequest,
)


class CodeInterpreter:
    def __init__(
        self,
        llm: Optional[BaseLanguageModel],
        additional_tools: list[BaseTool] = [],
        **kwargs,
    ) -> None:
        # params
        self.port: int = kwargs.get("port", 7801)
        self.max_iterations: int = kwargs.get("max_iterations", 10)
        self.verbose: bool = kwargs.get("verbose", False)
        self.log_handler: Callable = kwargs.get("log_handler", lambda x: x)

        # instances
        self.codebox: CustomLocalBox = CustomLocalBox(port=self.port)
        self.llm: BaseLanguageModel = None
        self.agent_executor: AgentExecutor = None
        self.tools: list[BaseTool] = create_tools(self._run_handler, additional_tools)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=ChatMessageHistory(),
        )

        self.update_llm(llm=llm)

        # contexts
        self.init_context()

    @property
    def session_id(self) -> Optional[UUID]:
        return self.codebox.session_id

    def init_context(self) -> Self:
        self.input_files: list[File] = []
        self.output_files: list[File] = []
        self.code_log: list[tuple[str, str]] = []
        return self

    def update_llm(self, llm: BaseLanguageModel) -> Self:
        self.llm: BaseLanguageModel = llm
        self.agent_executor: AgentExecutor = create_agent_executor(
            llm=llm,
            tools=self.tools,
            max_iterations=self.max_iterations,
            memory=self.memory,
            verbose=self.verbose,
        )
        return self

    def update_log_handler(self, log_handler: Callable) -> Self:
        self.log_handler: Callable = log_handler
        return self

    def _input_handler(self, request: UserRequest) -> None:
        """Callback function to handle user input."""
        if not request.files:
            return
        if not request.content:
            request.content = (
                "I uploaded, just text me back and confirm that you got the file(s)."
            )
        request.content += "\n**The user uploaded the following files: **\n"
        for file in request.files:
            self.input_files.append(file)
            request.content += f"[Attachment: {file.name}]\n"
            self.codebox.upload(file.name, file.content)
        request.content += "**File(s) are now available in the cwd. **\n"

    def _parse_output_files(self, code: str, output: CodeBoxOutput) -> str:
        if output.type == "image/png":
            filename = f"image-{uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            self.output_files.append(File(name=filename, content=file_buffer.read()))
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'", output.content
                ):
                    self.codebox.install(package.group(1))
                    return (
                        f"{package.group(1)} was missing but "
                        "got installed now. Please try again."
                    )
            else:
                # TODO: preanalyze error to optimize next code generation
                pass
            if self.verbose:
                print("Error:", output.content)

        elif modifications := get_file_modifications(code, self.llm):
            for filename in modifications:
                if filename in [file.name for file in self.input_files]:
                    continue
                fileb = self.codebox.download(filename)
                if not fileb.content:
                    continue
                file_buffer = BytesIO(fileb.content)
                file_buffer.name = filename
                self.output_files.append(
                    File(name=filename, content=file_buffer.read())
                )

        return output.content

    def _run_handler(self, code: str):
        """Run code in container and send the output to the user"""
        print("code:", code)
        self.log_handler(text=code, is_code=True)
        self.log_handler(text="コードを実行しています・・・")

        output: CodeBoxOutput = self.codebox.run(code)
        self.code_log.append((code, output.content))

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        self.log_handler(text="出力ファイルを抽出しています・・・")
        content: str = self._parse_output_files(code=code, output=output)
        return content

    def _output_handler(self, final_response: str) -> CodeInterpreterResponse:
        """Embed images in the response"""
        for file in self.output_files:
            if str(file.name) in final_response:
                # rm ![Any](file.name) from the response
                final_response = re.sub(r"\n\n!\[.*\]\(.*\)", "", final_response)

        if self.output_files and re.search(r"\n\[.*\]\(.*\)", final_response):
            try:
                final_response = remove_download_link(final_response, self.llm)
            except Exception as e:
                if self.verbose:
                    print("Error while removing download links:", e)

        output_files = self.output_files
        code_log = self.code_log
        self.output_files = []
        self.code_log = []

        return CodeInterpreterResponse(
            content=final_response, files=output_files, code_log=code_log
        )

    def generate_response_sync(
        self,
        user_msg: str,
        files: list[File] = [],
        streaming: bool = True,  # TODO: implement
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        user_request = UserRequest(content=user_msg, files=files)
        try:
            self._input_handler(user_request)
            assert self.agent_executor, "Session not initialized."
            response = self.agent_executor.run(input=user_request.content)
            return self._output_handler(response)
        except Exception as e:
            if self.verbose:
                traceback.print_exc()
            return CodeInterpreterResponse(
                content="Something error went while generating your response: "
                f"{e.__class__.__name__}  - {e} "
                "please restart the session."
            )

    def is_running(self) -> bool:
        return self.codebox.status() == "running"

    def start(self) -> CodeBoxStatus:
        return self.codebox.start()

    def stop(self) -> CodeBoxStatus:
        return self.codebox.stop()

    def reconnect(self) -> None:
        self.init_context()
        self.codebox.connect()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
