from os import getenv
from typing import Optional

from langchain.chat_models import AzureChatOpenAI, ChatAnthropic, ChatOpenAI
from langchain.chat_models.base import BaseChatModel


def buildup_llm(
    model: str = "gpt-4", openai_api_key: Optional[str] = None, **kwargs
) -> BaseChatModel:
    if "gpt" in model:
        openai_api_key = openai_api_key or getenv("OPENAI_API_KEY", None)
        if openai_api_key is None:
            raise ValueError(
                "OpenAI API key missing. Set OPENAI_API_KEY env variable "
                "or pass `openai_api_key` to session."
            )

        openai_api_version = getenv("OPENAI_API_VERSION")
        openai_api_base = getenv("OPENAI_API_BASE")
        deployment_name = getenv("DEPLOYMENT_NAME")
        openapi_type = getenv("OPENAI_API_TYPE")

        if (
            openapi_type == "azure"
            and openai_api_version
            and openai_api_base
            and deployment_name
        ):
            return AzureChatOpenAI(
                temperature=0.03,
                openai_api_base=openai_api_base,
                openai_api_version=openai_api_version,
                deployment_name=deployment_name,
                openai_api_key=openai_api_key,
                max_retries=3,
                request_timeout=60 * 3,
            )  # type: ignore
        else:
            return ChatOpenAI(
                temperature=0.03,
                model=model,
                openai_api_key=openai_api_key,
                max_retries=3,
                request_timeout=60 * 3,
            )  # type: ignore
    elif "claude" in model:
        return ChatAnthropic(model=model)
    else:
        raise ValueError(f"Unknown model: {model} (expected gpt or claude model)")
