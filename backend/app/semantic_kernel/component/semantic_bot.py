import asyncio
import os

import openai
from dotenv import load_dotenv

from app.base.component.logger import Logger
from app.semantic_kernel.component.semantic_memory import (
    SemanticMemory,
    SemanticMemorySummarizer,
)

g_logger = Logger(logger_name="app")
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class SemanticBot(object):
    def __init__(self) -> None:
        self.memory = SemanticMemory()

    def _llm(self, **params):
        response = openai.ChatCompletion.create(**params)["choices"][0]["message"][
            "content"
        ]
        return response

    async def do_chat(
        self,
        history: list[tuple[str, str]],
        model_name: str,
        temperature_percent: int,  # in [0, 200]
        max_tokens: int = 1024,
    ) -> list[tuple[str, str]]:
        query: str = history[-1][0]
        temperature: float = temperature_percent / 100

        task = asyncio.create_task(self.memory.search(query=query))
        results = await task
        previous_context = self.summarize(
            query, search_results=results, model_name=model_name
        )

        prompt = f"""必要に応じて`文脈:`を参考に、以下の`質問:`について正確な回答をするようにステップバイステップでロジックツリー形式で回答してください。
質問:
```
{query}
```

文脈:
```
{previous_context if previous_context else "なし"}
```
"""

        answer = self.exec_llm(
            query=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        assert answer
        history[-1][1] = answer  # may be changed url to href

        context = f"Q: {query}\nA: {answer}"
        await self.memory.append(context)

        return history

    def summarize(
        self, query: str, search_results: list, model_name: str = "gpt-3.5-turbo"
    ):
        template = (
            f"以下の文を 質問 `{query}` の意図に正確に答えるように500文字以内になるようにステップバイステップで要点をまとめて洗練させてください。"  # noqa
            + """
```
{{$input}}
```
"""
        )
        summarizer = SemanticMemorySummarizer(template=template, model_name=model_name)
        summarized: str = summarizer.summarize_results(search_results)
        return summarized

    def exec_llm(
        self,
        query: str,
        model_name: str,
        temperature: float,  # in [0, 2.0]
        max_tokens: int = 1024,
    ):
        messages = [{"role": "user", "content": query}]
        params = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        answer = self._llm(**params)

        return answer
