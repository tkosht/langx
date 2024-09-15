import os

import openai
from dotenv import load_dotenv
from semantic_kernel import SKContext
from semantic_kernel.skill_definition import sk_function, sk_function_context_parameter

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class Answer:
    skill_name = "AnswerSkill"

    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens

    def _params(self, messages: list[dict]):
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        return params

    def _llm(self, message: str) -> str:
        params = self._params(messages=[{"role": "user", "content": message}])
        response: str = openai.ChatCompletion.create(**params)["choices"][0]["message"][
            "content"
        ]
        return response

    @sk_function(
        name="answer_skill",
        description="任意のテキストに対して、大規模言語モデルを使用して要約やまとめなどの回答を生成する際に利用します。過去の依頼や回答も含めること。",  # 最後のツール・スキルとして必ず指定すること
        input_description="arbitrary text like user input or previous tool's/skill's output",
    )
    @sk_function_context_parameter(
        name="query",
        description="どのような回答を作成したいかを指定します",
    )
    @sk_function_context_parameter(
        name="past_context",
        description="過去のやり取りなどの文脈(文章もOK)を指定します",
    )
    def make_answer(self, context: SKContext) -> str:
        input_text = context["input"]
        try:
            past_context = context["past_context"]
        except Exception:
            past_context = "なし"
        try:
            query = context["query"]
        except Exception:
            query = "`入力テキスト`を人間が読みやすいようにまとめます。"
        query += " 特に、自ら文章を創作しないようにし、情報源のリンクが明記されている場合は省略せずMarkdown形式で表現すること"
        print(f"Answer.make_answer: {input_text=}, {query=}, {past_context=}")

        message = f"""以下の`テキスト`に対して、`ゴール` を満たすように 以下の`指示`に従い文章を生成してください。`指示`がない場合は`入力テキスト`に従ってください

# 入力テキスト
```
{input_text}
```

# 指示
```
{query}
```

# 過去の文脈
```
{past_context}
```
"""

        response: str = self._llm(message=message)
        return response
