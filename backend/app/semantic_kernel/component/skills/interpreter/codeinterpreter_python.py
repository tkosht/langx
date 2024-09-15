import base64
import json

from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter

from app.codeinterpreter.component.interpreter import (
    CodeInterpreter,
    CodeInterpreterResponse,
)


class CodeInterpeterPython(object):
    def __init__(self, port=7890) -> None:
        self.session = CodeInterpreter(port=port)

    def start(self) -> None:
        self.session.start()

    async def astart(self) -> None:
        await self.session.astart()

    @sk_function(
        name="code_interpreter_in_python",
        description="""グラフ化やデータ加工やモデル作成、処理結果の変数への保持等のデータ分析タスクの結果を返します。この結果は直接ユーザに返すことができます。""",
        input_description="user input or previous output, other args are ignored",
    )
    async def run(self, context: SKContext) -> str:
        request: str = context["input"]
        print(f"{request=}")
        response: CodeInterpreterResponse = await self.session.generate_response(
            request
        )
        resdic = dict(
            text=response.content,
            images=[base64.b64encode(fl.content).decode() for fl in response.files],
        )
        res = json.dumps(resdic)
        return res

    async def astop(self):
        await self.session.astop()

    def stop(self):
        self.session.stop()
