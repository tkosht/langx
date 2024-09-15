"""
このコードは `Dominic Bäumer` 氏 のプロジェクト(https://github.com/shroominic/codeinterpreter-api.git)を参考に作成しました。

オリジナルのライセンス:
- MIT License
"""

import json
from typing import List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.schema import AIMessage, OutputParserException

from app.codeinterpreter.component.llm.prompts import (
    determine_modifications_prompt,
    remove_dl_link_prompt,
)


def get_file_modifications(
    code: str,
    llm: BaseLanguageModel,
    retry: int = 2,
) -> Optional[List[str]]:
    if retry < 1:
        return None

    prompt = determine_modifications_prompt.format(code=code)

    result = llm.predict(prompt, stop="```")

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        result = ""
    if not result or not isinstance(result, dict) or "modifications" not in result:
        return get_file_modifications(code, llm, retry=retry - 1)
    return result["modifications"]


async def aget_file_modifications(
    code: str,
    llm: BaseLanguageModel,
    retry: int = 2,
) -> Optional[List[str]]:
    if retry < 1:
        return None

    prompt = determine_modifications_prompt.format(code=code)

    result = await llm.apredict(prompt, stop="```")

    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        result = ""
    if not result or not isinstance(result, dict) or "modifications" not in result:
        return await aget_file_modifications(code, llm, retry=retry - 1)
    return result["modifications"]


def remove_download_link(
    input_response: str,
    llm: BaseLanguageModel,
) -> str:
    messages = remove_dl_link_prompt.format_prompt(
        input_response=input_response
    ).to_messages()
    message = llm.predict_messages(messages)

    if not isinstance(message, AIMessage):
        raise OutputParserException("Expected an AIMessage")

    return message.content


async def aremove_download_link(
    input_response: str,
    llm: BaseLanguageModel,
) -> str:
    messages = remove_dl_link_prompt.format_prompt(
        input_response=input_response
    ).to_messages()
    message = await llm.apredict_messages(messages)

    if not isinstance(message, AIMessage):
        raise OutputParserException("Expected an AIMessage")

    return message.content
