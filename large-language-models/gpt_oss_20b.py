#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub", "llama-cpp-python", "muna"]
# ///

from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from muna import compile, Parameter, Sandbox
from muna.beta import Annotations, LlamaCppInferenceMetadata
from muna.beta.openai import ChatCompletionChunk, Message
from typing import Annotated, Iterator

# Load gpt-oss-20b
model_path = hf_hub_download(
    repo_id="ggml-org/gpt-oss-20b-GGUF",
    filename="gpt-oss-20b-mxfp4.gguf"
)
model = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False)

@compile(
    sandbox=Sandbox().pip_install("huggingface_hub", "llama-cpp-python"),
    metadata=[
        LlamaCppInferenceMetadata(
            model=model,
            backends=["cuda"]
        )
    ]
)
def gpt_oss_20b(
    messages: Annotated[
        list[Message],
        Parameter.Generic(description="Messages comprising the chat conversation so far.")
    ],
    *,
    response_format: Annotated[
        dict,
        Annotations.ResponseFormat(description="Format specification to use in response.")
    ]=None,
    max_output_tokens: Annotated[
        int,
        Annotations.MaxOutputTokens(description="Maximum number of tokens in the response.")
    ]=5_000,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature (randomness).",
        min=0.0,
        max=2.0
    )]=1.0,
    top_p: Annotated[float, Annotations.TopP(
        description="Sampling probability for token selection.",
        min=0.0,
        max=1.0
    )]=1.0,
    presence_penalty: Annotated[float, Annotations.PresencePenalty(
        description="Penalty for tokens that have already appeared.",
        min=0.0,
        max=2.0
    )]=0.0,
    frequency_penalty: Annotated[float, Annotations.FrequencyPenalty(
        description="Penalty for tokens based on their frequency.",
        min=0.0,
        max=2.0
    )]=0.0
) -> Iterator[ChatCompletionChunk]:
    """
    Create chat conversations with OpenAI gpt-oss-20b.
    """
    for chunk in model.create_chat_completion(
        messages=messages,
        max_tokens=max_output_tokens,
        stream=True,
        response_format=response_format,
        top_p=top_p,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    ):
        yield chunk

if __name__ == "__main__":
    stream = gpt_oss_20b([
        Message(role="system", content="You are an AI assistant that provides very brief answers."),
        Message(role="user", content="What is the capital of France.")
    ])
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta.get("content"), end="")
    print()