import os

import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import (  # AzureTextCompletion,; AzureTextEmbedding,; ; OpenAITextCompletion,   # noqa
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from typing_extensions import Any, Self

from app.semantic_kernel.component.planner import CustomPlanner
from app.semantic_kernel.component.skills.interpreter.codeinterpreter_python import (
    CodeInterpeterPython,
)


class SimpleRunner(object):
    def __init__(
        self,
        planner: CustomPlanner = CustomPlanner(),
        skill_dir=None,
        model_name: str = "gpt-3.5-turbo",
    ) -> None:
        self.planner: CustomPlanner = planner
        self.skill_dir = skill_dir
        self.model_name = model_name

        self.kernel: sk.Kernel = None
        self.skills: list[dict[str, SKFunctionBase]] = []
        self.memory_store = sk.memory.VolatileMemoryStore()
        self.code_interpreter: CodeInterpeterPython = None

        self.setup_kernel().setup_skills()

    def set_planner(self, planner: CustomPlanner) -> Self:
        self.planner = planner
        return self

    def setup_kernel(self) -> Self:
        kernel = sk.Kernel()
        load_dotenv()

        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")

        kernel.add_chat_service(
            "gpt", OpenAIChatCompletion(self.model_name, api_key, org_id)
        )
        kernel.add_text_embedding_generation_service(
            "ada", OpenAITextEmbedding("text-embedding-ada-002", api_key, org_id)
        )
        kernel.register_memory_store(memory_store=self.memory_store)
        self.kernel = kernel

        return self

    def update_kernel_model(self, model_name: str = None) -> Self:
        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")

        try:
            self.kernel.remove_chat_service("gpt")
        except Exception:
            pass
        self.kernel.add_chat_service(
            "gpt", OpenAIChatCompletion(model_name, api_key, org_id)
        )
        return self

    def setup_skills(self, model_name: str = "gpt-3.5-turbo") -> Self:
        # from semantic_kernel.core_skills.http_skill import HttpSkill

        # from app.semantic_kernel.component.skills.answer.native_function import Answer
        # from app.semantic_kernel.component.skills.search.search_local import SearchLocal
        # from app.semantic_kernel.component.skills.search.search_web import SearchWeb

        # self.skills.append(self.kernel.import_skill(SearchLocal(), "SearchLocal"))
        # self.skills.append(self.kernel.import_skill(SearchWeb(), "SearchWeb"))
        # self.skills.append(self.kernel.import_skill(HttpSkill(), "HttpSkill"))
        # self.skills.append(
        #     self.kernel.import_skill(Answer(model_name=model_name), "Answer")
        # )
        # self.skills.append(
        #     self.kernel.import_native_skill_from_directory(skill_dir, "math")
        # )
        self.code_interpreter = CodeInterpeterPython()
        self.skills.append(
            self.kernel.import_skill(self.code_interpreter, "CodeInterpeterPython")
        )
        return self

    async def do_run(self, user_query: str, n_retries: int = 3) -> str:
        meta_order = "以下の`- ユーザの依頼`について過去のやり取り(文脈)も踏まえて実行プランを作成してください。"  # noqa
        # constraint = "最後に Answer を必ず使ってください。ステップバイステップでプランを作成してください。"
        constraint = "極力、データサイエンスやデータ分析タスクとして扱ってください。そして無駄がないように最も少ない `subtasks` で実現してください。特に、_GLOBAL_FUNCTIONS_.* は絶対に使わないでください。"
        input_query = f"""[GOALここから]
{meta_order}

- ユーザの依頼
(((
{user_query}
)))

- 制約
(((
{constraint}
)))
[GOALここまで]
"""

        for _ in range(n_retries):
            try:
                input_query = input_query.replace("\\x", "\\\\x")
                print("-" * 50)
                print("input_query:", input_query)
                generated_plan: sk.SKContext = await self.do_plan(
                    input_query=input_query
                )
                print(f"generated_plan: {generated_plan.result}")
                print("-" * 25)
                response = await self.do_execute(generated_plan)
                break
            except Exception as e:
                input_query = f"""[GOALここから]
{meta_order}

- ユーザの依頼
(((
{user_query}
)))

- 制約
(((
{constraint}
)))

- あなたは、直前に以下のようなプランを作成し実行しました
(((
{generated_plan.result}
)))

- しかし以下のようなエラーが発生しました
(((
{e}
)))

- エラーが起きないように確実に対処するように、ステップバイステップでプランの見直しを改めて検討してください
[GOALここまで]
"""

                print(input_query)
                response = f"プランの作成と実行に失敗しました\n\n{user_query}\n\n{e}"
                continue
        print(response)
        return response

    async def do_plan(
        self, input_query: str, prompt: str = None, n_retries: int = 3
    ) -> sk.SKContext:
        params = dict(goal=input_query, kernel=self.kernel)
        _params = params.copy()
        if prompt:
            _params.update(dict(prompt=prompt))
        generated_plan: sk.SKContext = await self.planner.create_plan_async(**_params)
        return generated_plan

    async def do_execute(self, generated_plan: sk.SKContext) -> Any:
        response = await self.planner.execute_plan_async(generated_plan, self.kernel)
        return response

    def start(self) -> Self:
        self.code_interpreter.start()
        return self

    async def astart(self) -> Self:
        await self.code_interpreter.astart()
        return self

    async def astop(self) -> Self:
        await self.code_interpreter.astop()
        return self

    async def stop(self) -> Self:
        await self.code_interpreter.stop()
        return self
