import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

logger = logging.getLogger(__name__)


class LLM:
    def __init__(
        self,
        context_window = 2048,
        max_new_tokens = 256,
        tokenizer_name = "denisgr04/guap",
        model_name = "denisgr04/guap",
        model = None,
        tokenizer = None,
        device_map = "auto",
        tokenizer_kwargs = {},
        model_kwargs = {},
        generate_kwargs = {}
    ):

        self._model = model or AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, **model_kwargs)

        config_dict = self._model.config.to_dict()
        model_context_window = int(
            config_dict.get("max_position_embeddings", context_window)
        )
        if model_context_window and model_context_window < context_window:
            context_window = model_context_window

        tokenizer_kwargs = tokenizer_kwargs or {}
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = context_window

        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            tokenizer_name, **tokenizer_kwargs
        )

        self.context_window=context_window
        self.max_new_tokens=max_new_tokens
        self.tokenizer_name=tokenizer_name
        self.model_name=model_name
        self.device_map=device_map
        self.tokenizer_kwargs=tokenizer_kwargs or {}
        self.model_kwargs=model_kwargs or {}
        self.generate_kwargs=generate_kwargs or {}

    def complete(self, prompt: str, tokenizer_in_kwargs={},
                 generate_kwargs={},
                 tokenizer_out_kwargs={}) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", 
                                 **tokenizer_in_kwargs)
        inputs = inputs.to(self._model.device)
        tokens = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            **self.generate_kwargs,
            **generate_kwargs
        )
        completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
        completion = self._tokenizer.decode(completion_tokens,
                                            skip_special_tokens=True,
                                            **tokenizer_out_kwargs)
        return completion
