import math

from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.skill_definition import (sk_function,
                                              sk_function_context_parameter)


class MathPlugin:
    # sqrt method
    @sk_function(
        description="Takes the square root of a number",
        name="square_root",
        input_description="The value to take the square root of",
    )
    def square_root(self, number: str) -> str:
        print(f"MathPlugin.add: {number=}")
        return str(math.sqrt(float(number)))

    # add method
    @sk_function(
        description="Adds two numbers together",
        name="add",
    )
    @sk_function_context_parameter(
        name="input",
        description="The first number to add",
    )
    @sk_function_context_parameter(
        name="number2",
        description="The second number to add",
    )
    def add(self, context: SKContext) -> str:
        number1 = float(context["input"])
        number2 = float(context["number2"])
        print(f"MathPlugin.add: {number1=}, {number2=}")
        return str(number1 + number2)
