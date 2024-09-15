import asyncio
import itertools
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional, Tuple, Union

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler,
    ExLlamaV2StreamingGenerator,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk


class ChatExllamaV2Model(BaseChatModel):
    """exllamav2 models.

    To use, you should have the ``exllamav2`` python package installed.

    Example:
        .. code-block:: python

    """

    exllama_config: ExLlamaV2Config
    """ ExLlamaV2 config """
    exllama_model: ExLlamaV2
    """ ExLlamaV2 pretrained Model """
    exllama_tokenizer: ExLlamaV2Tokenizer
    """ ExLlamaV2 Tokenizer """
    exllama_cache: Union[ExLlamaV2Cache, ExLlamaV2Cache_8bit]
    """ ExLLamaV2 Cache """

    # メッセージテンプレート
    human_message_template: str = "USER: {}\n"
    ai_message_template: str = "ASSISTANT: {}"
    system_message_template: str = "{}"

    do_sample: bool = True
    max_new_tokens: int = 64
    repetition_penalty: float = 1.1
    temperature: float = 1
    top_k: int = 50
    top_p: float = 0.8

    prompt_line_separator: str = "\n"

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        cache_8bit: bool = False,
        cache_max_seq_len: int = -1,
        tokenizer_force_json=False,
        **kwargs: Any,
    ) -> "ChatExllamaV2Model":
        """Construct the exllamav2 model and tokenzier pipeline object from model_id"""

        # Initialize config
        config = ExLlamaV2Config()
        config.model_dir = model_dir
        config.prepare()

        # Initialize model
        model = ExLlamaV2(config)

        # Initialize tokenizer
        tokenizer = ExLlamaV2Tokenizer(config, force_json=tokenizer_force_json)

        # cache
        cache = None
        if cache_8bit:
            cache = ExLlamaV2Cache_8bit(
                model, lazy=not model.loaded, max_seq_len=cache_max_seq_len
            )
        else:
            cache = ExLlamaV2Cache(
                model, lazy=not model.loaded, max_seq_len=cache_max_seq_len
            )

        # load model
        model.load_autosplit(cache)

        return cls(
            exllama_config=config,
            exllama_model=model,
            exllama_tokenizer=tokenizer,
            exllama_cache=cache,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "ChatExllamaV2Model"

    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"{self.prompt_line_separator}{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = self.human_message_template.format(message.content)
        elif isinstance(message, AIMessage):
            message_text = self.ai_message_template.format(message.content)
        elif isinstance(message, SystemMessage):
            message_text = self.system_message_template.format(message.content)
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return self.prompt_line_separator.join(
            [self._format_message_as_text(message) for message in messages]
        )

    def _generate_streamer(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ExLlamaV2StreamingGenerator:
        prompt = self._format_messages_as_text(messages)
        _stop = stop or []

        generator = ExLlamaV2StreamingGenerator(
            self.exllama_model, self.exllama_cache, self.exllama_tokenizer
        )

        # Settings
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = self.temperature
        settings.top_k = self.top_k
        settings.top_p = self.top_p
        settings.token_repetition_penalty = self.repetition_penalty

        # Prompt
        # add_bos/add_eosの指定はプロンプトの種類に応じて考慮が必要かも。
        input_ids = self.exllama_tokenizer.encode(
            prompt, add_bos=True, add_eos=False, encode_special_tokens=True
        )

        prompt_tokens = input_ids.shape[-1]

        # stopトークンリストの作成
        _stop_ids = [self.exllama_tokenizer.encode(s)[0] for s in _stop]
        _stop_ids = [_id.tolist() for _id in _stop_ids if _id.shape[-1] > 0]
        _stop_ids = list(itertools.chain.from_iterable(_stop_ids))

        generator.set_stop_conditions(_stop_ids + [self.exllama_tokenizer.eos_token_id])
        generator.begin_stream(input_ids, settings)

        return generator

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        streamer = self._generate_streamer(messages, stop, run_manager, **kwargs)

        texts = []
        generated_tokens = 0

        while True:
            chunk, eos, _ = streamer.stream()
            generated_tokens += 1
            texts.append(chunk)
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk,
                    verbose=self.verbose,
                )

            if eos or generated_tokens == self.max_new_tokens:
                break

        chat_generation = ChatGeneration(message=AIMessage(content="".join(texts)))
        return ChatResult(
            generations=[chat_generation],
            llm_output={"completion_tokens": generated_tokens},
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        streamer = self._generate_streamer(messages, stop, run_manager, **kwargs)

        texts = []
        generated_tokens = 0

        while True:
            chunk, eos, _ = streamer.stream()
            generated_tokens += 1
            texts.append(chunk)
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk,
                    verbose=self.verbose,
                )

            await asyncio.sleep(0)
            if eos or generated_tokens == self.max_new_tokens:
                break

        chat_generation = ChatGeneration(message=AIMessage(content="".join(texts)))
        return ChatResult(
            generations=[chat_generation],
            llm_output={"completion_tokens": generated_tokens},
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        streamer = self._generate_streamer(messages, stop, run_manager, **kwargs)
        generated_tokens = 0

        while True:
            chunk, eos, _ = streamer.stream()
            generated_tokens += 1
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk,
                    verbose=self.verbose,
                )
            if eos or generated_tokens == self.max_new_tokens:
                break

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        streamer = self._generate_streamer(messages, stop, run_manager, **kwargs)
        generated_tokens = 0

        while True:
            chunk, eos, _ = streamer.stream()
            generated_tokens += 1
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk,
                    verbose=self.verbose,
                )
            await asyncio.sleep(0)
            if eos or generated_tokens == self.max_new_tokens:
                break

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "config": self.exllama_config,
            "model": self.exllama_model,
            "tokenizer": self.exllama_tokenizer,
        }
