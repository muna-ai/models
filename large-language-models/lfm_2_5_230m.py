#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "llama-cpp-python", "muna"]
# ///

from huggingface_hub import hf_hub_download
from json import dumps
from llama_cpp import Llama, LlamaGrammar
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, LlamaCppInferenceMetadata
from muna.beta.openai import ChatCompletion, Message
from pydantic import BaseModel
from typing import Annotated

model_path = hf_hub_download(
    "LiquidAI/LFM2.5-230M-GGUF",
    "LFM2.5-230M-Q8_0.gguf"
)
model = Llama(model_path=model_path, verbose=False)

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "llama-cpp-python"),
    metadata=[
        LlamaCppInferenceMetadata(model=model)
    ]
)
def lfm_2_5_230m(
    messages: Annotated[
        list[Message],
        Parameter.Generic(description="Messages comprising the chat conversation so far.")
    ],
    *,
    response_format: Annotated[
        dict,
        Annotations.ResponseFormat(description="Response format.")
    ] = None,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=4096
    )]=4096,
) -> ChatCompletion:
    """
    Parse structured outputs with LFM2 350M Extract.
    """
    schema = response_format["json_schema"]["schema"]
    grammar = LlamaGrammar.from_json_schema(dumps(schema))
    return model.create_chat_completion(
        messages=messages,
        grammar=grammar,
        max_tokens=max_output_tokens
    )

if __name__ == "__main__":
    class Pet(BaseModel):
        name: str
        legs: int
    completion = lfm_2_5_230m(
        [
            { "role": "system", "content": "You are an AI assistant that parses structured output from input text." },
            { "role": "user", "content": "My name is Yusuf and I have four legs." },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "pet",
                "strict": True,
                "schema": Pet.model_json_schema()
            }
        }
    )
    print(completion)