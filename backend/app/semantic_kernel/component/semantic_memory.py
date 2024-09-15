import os
import re
from dataclasses import dataclass

import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import (  # AzureTextCompletion,; AzureTextEmbedding,; ; OpenAITextCompletion,   # noqa
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.memory.memory_query_result import MemoryQueryResult
from typing_extensions import Self

from app.base.component.logger import Logger
from app.base.component.ulid import build_ulid

load_dotenv()
g_logger = Logger()


@dataclass
class SemanticMemorySummarizer(object):
    template: str
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 2000
    chunk_size: int = 1024

    def __post_init__(self) -> None:
        self.kernel: sk.Kernel = None
        self.setup_kernel(model_name=self.model_name)
        assert self.kernel is not None

        self.summarizer = self.kernel.create_semantic_function(
            self.template, max_tokens=self.max_tokens, temperature=0.0, top_p=0.8
        )

    def setup_kernel(self, model_name: str = "gpt-3.5-turbo") -> Self:
        kernel = sk.Kernel()

        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")

        kernel.add_chat_service(
            "gpt", OpenAIChatCompletion(model_name, api_key, org_id)
        )
        self.kernel = kernel
        return self

    def summarize_results(self, results: list[MemoryQueryResult]) -> str:
        summarized: str = ""
        for r in results:
            g_logger.info(f"{r.additional_metadata=} / {r.relevance}")
            text = re.sub(r"\n+", "\n", r.text)
            text = re.sub(r"\s\s+", " ", text)
            summarized = summarized + "\n" + text
            summarized: str = self.summarize(summarized)
            g_logger.info(f"{text=}")
            g_logger.info(f"{summarized=}")
        return summarized

    def summarize(self, text: str) -> str:
        summarized: str = ""
        for idx in range(0, len(text), self.chunk_size):
            response = self.summarizer(
                summarized + "\n" + text[idx : idx + self.chunk_size]
            )
            summarized: str = response.result
        return summarized


class SemanticMemory(object):
    def __init__(self) -> None:
        self.kernel: sk.Kernel = None

        self.setup_kernel()
        assert self.kernel is not None

    def setup_kernel(self, embedding_model: str = "text-embedding-ada-002") -> Self:
        kernel = sk.Kernel()

        api_key = os.environ.get("OPENAI_API_KEY")
        org_id = os.environ.get("OPENAI_ORG_ID")

        kernel.add_text_embedding_generation_service(
            "ada", OpenAITextEmbedding(embedding_model, api_key, org_id)
        )

        kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
        self.kernel = kernel
        return self

    async def append(
        self,
        text: str,
        memory_key: str = "memory",
        memory_id: str = None,
        additional_metadata: str = None,
    ) -> Self:
        if memory_id is None:
            memory_id = build_ulid(prefix="INF")
        await self.kernel.memory.save_information_async(
            memory_key,
            id=memory_id,
            text=text,
            additional_metadata=additional_metadata,
        )
        return self

    def clear(self) -> Self:
        # recreate the memory store
        self.kernel.memory = None
        self.kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
        assert self.kernel.memory is not None
        return self

    async def search(
        self,
        query: str,
        memory_key: str = "memory",
        limit: int = 3,
        min_relevance: float = 0.7,
    ) -> list[MemoryQueryResult]:
        self.query = query
        results = await self.kernel.memory.search_async(
            memory_key, query, limit=limit, min_relevance_score=min_relevance
        )
        return results


async def _main():
    import requests
    from bs4 import BeautifulSoup

    g_logger.info("Start")

    urls = [
        "https://xtech.nikkei.com/atcl/nxt/column/18/00001/08164/",
        "https://xtech.nikkei.com/atcl/nxt/column/18/00682/062700127/",
        "https://xtech.nikkei.com/atcl/nxt/column/18/00001/08158/",
        "https://xtech.nikkei.com/atcl/nxt/column/18/02504/062700004/",
        "https://xtech.nikkei.com/atcl/nxt/column/18/02498/061500001/",
        "https://xtech.nikkei.com/atcl/nxt/column/18/02504/062600003/",
    ]

    smm = SemanticMemory()

    g_logger.info("processing to register web page chunks")
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "lxml")

        memory_chunk_size = 1024
        for idx in range(0, len(soup.text), memory_chunk_size):
            await smm.append(
                text=soup.text[idx : idx + memory_chunk_size], additional_metadata=url
            )

    query = "LLMについて教えて"
    task = asyncio.create_task(smm.search(query=query))
    results = await task

    g_logger.info("processing to summarize")

    template = (
        f"以下の文を 質問 `{query}` の意図に正確に答えるように500文字以内になるようにステップバイステップで要点をまとめて洗練させてください。"  # noqa
        + """
```
{{$input}}
```
"""
    )
    summarizer = SemanticMemorySummarizer(template=template)
    summarizer.summarize_results(results)

    g_logger.info("End")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
