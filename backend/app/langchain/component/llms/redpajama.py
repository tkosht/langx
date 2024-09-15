from typing import Any, List, Mapping, Optional

import torch
import transformers
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

MIN_TRANSFORMERS_VERSION = "4.25.1"

# check transformers version
assert (
    transformers.__version__ >= MIN_TRANSFORMERS_VERSION
), f"Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher."

# init
model_ident = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ident)
g_model = AutoModelForCausalLM.from_pretrained(model_ident, torch_dtype=torch.float16)


# NOTE: just implement
# TODO: to test
class RedPajamaLLM(LLM):
    device: torch.device
    temperature: float
    n_tokens: int
    model_name: str = model_ident

    def __init__(self):
        self.model = g_model.to(self.device)

    @property
    def _llm_type(self) -> str:
        return "RedPajama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        print("-" * 100)
        print(f"{prompt=}")

        inputs = tokenizer(prompt, return_tensors="pt").to(g_model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = g_model.generate(
            **inputs,
            max_new_tokens=128,
            min_length=1,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.7,
            top_k=50,
            return_dict_in_generate=True,
            repetition_penalty=1.11,
            length_penalty=0.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        token = outputs.sequences[0, input_length:]
        output_str = tokenizer.decode(token)

        print("-" * 100)
        print(f"{output_str=}")

        return output_str

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"temperature": self.temperature, "n_tokens": self.n_tokens}
