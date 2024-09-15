import json

import regex
import semantic_kernel as sk
from semantic_kernel.planning.basic_planner import PROMPT, BasicPlanner


# NOTE: cf. python3.10/dist-packages/semantic_kernel/planning/basic_planner.py
class CustomPlanner(BasicPlanner):
    def __init__(self, max_tokens: int = 1000, temperature: float = 0.0) -> None:
        super().__init__()

        self.max_tokens = max_tokens
        self.temperature = temperature

    async def create_plan_async(
        self,
        goal: str,
        kernel: sk.Kernel,
        prompt: str = PROMPT,
    ) -> sk.SKContext:
        planner = kernel.create_semantic_function(
            prompt, max_tokens=self.max_tokens, temperature=self.temperature
        )

        available_functions_string = self._create_available_functions_string(kernel)

        context = kernel.create_new_context()
        context["goal"] = goal
        context["available_functions"] = available_functions_string
        generated_plan = await planner.invoke_async(context=context)
        return generated_plan

    async def execute_plan_async(
        self, generated_plan: sk.SKContext, kernel: sk.Kernel
    ) -> str:
        json_regex = r"\{(?:[^{}]|(?R))*\}"
        generated_plan_string = regex.search(json_regex, generated_plan.result).group()
        generated_plan = json.loads(generated_plan_string)

        context = kernel.create_new_context()
        context["input"] = generated_plan["input"]
        subtasks = generated_plan["subtasks"]

        for subtask in subtasks:
            skill_name, function_name = subtask["function"].split(".")
            sk_function = kernel.skills.get_function(skill_name, function_name)

            args = subtask.get("args", None)
            if args:
                for key, value in args.items():
                    if isinstance(value, str):
                        context[key] = value
                    else:
                        print(f"WARNING not a string value: {key}={value}")
            output = await sk_function.invoke_async(context=context)

            context["input"] = output.result

        return output.result
