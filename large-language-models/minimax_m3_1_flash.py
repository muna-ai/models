#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "accelerate", "muna", "sglang==0.5.21",
#   "tokenizers==0.22.2", "torch", "transformers==5.15.0"
# ]
# ///

# `sglang` is used for its streaming reasoning / tool-call parsers only; the
# MiniMax-M3 parsers first ship in v0.5.21. PyPI ships Linux-only wheels for
# it; on macOS install it from the tag without the Rust extensions (which the
# parsers never import):
#
#   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0
#   SGLANG_BUILD_RUST_EXTS=none pip install --no-deps --no-build-isolation \
#     "git+https://github.com/sgl-project/sglang.git@v0.5.21#subdirectory=python"
#   pip install msgspec orjson pybase64 pyzmq starlette IPython \
#     openai==2.6.1 partial_json_parser xgrammar

from __future__ import annotations
from accelerate import init_empty_weights
from muna import compile, BatchConfig, Sandbox
from muna.beta import Annotations, KVRoutingMetadata, TorchToSGLangInferenceMetadata
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, ChoiceDeltaToolCall,
    DeltaMessage, Message, StreamChoice
)
from os import environ
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from time import time
from tokenizers.decoders import DecodeStream
from torch import bfloat16
from torch.nn import Embedding
from transformers import AutoTokenizer, GenerationMixin, PreTrainedModel
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator, Literal
from uuid import uuid4

# Load the MiniMax-M3.1 Flash text backbone.
# Transformers has no MiniMax-M3 model class and the checkpoint's remote code
# covers only the VL config, so the model cannot be instantiated here. The
# compiler only needs a model carrying the text config, so define a stub. The
# engine serves text only (the VL wrapper's vision tower is not loaded).
CHECKPOINT = "MiniMaxAI/MiniMax-M3.1-Flash"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

class MiniMaxM3TextConfig(PreTrainedConfig):
    model_type = "minimax_m3_text"

class MiniMaxM3ForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMaxM3TextConfig
    _supports_attention_backend = True

    def __init__(self, config: MiniMaxM3TextConfig):
        super().__init__(config)
        # The batching manager reads the model device and dtype off its parameters
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, dtype=bfloat16)
        self.post_init()

config_dict, _ = PreTrainedConfig.get_config_dict(CHECKPOINT)
text_config = MiniMaxM3TextConfig.from_dict(config_dict["text_config"])
text_config._name_or_path = CHECKPOINT
with init_empty_weights():
    model = MiniMaxM3ForCausalLM(text_config)

# Resolve the reasoning marker token. The template's generation prompt is
# adaptive (no think prefix): the model decides per turn whether to open
# `<mm:think>`. Starting the reply with `</mm:think>` disables thinking.
THINK_CLOSE = "</mm:think>"
STOP_TOKENS = [tokenizer.eos_token_id]

# Engine limits
MAX_TOTAL_TOKENS = 1_048_576

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=STOP_TOKENS,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
)
batching_config = ContinuousBatchingConfig(
    per_request_processors=True,
    use_cuda_graph=True,
    max_memory_percent=0.9,
)
manager = model.init_continuous_batching(
    generation_config=generation_config,
    continuous_batching_config=batching_config,
)

# Define a tokenization function
# This function is used for both inference and KV-aware routing
def _tokenize(
    messages: list[Message],
    tools: list[dict] | None = None,
    reasoning_effort: str | None = None
) -> list[int]:
    # The template maps `system` and `developer` both onto its developer
    # prompt, so messages pass through unchanged. Tool-call arguments arrive
    # as JSON strings; the runtime decodes them into mappings before rendering
    # (as upstream SGLang does), which the template needs to render them as
    # XML parameters.
    # Map the OpenAI reasoning effort onto the template's `<effort>` tag, as
    # upstream `_normalize_minimax_m3_request` (serving_chat.py) does: "none"
    # and "minimal" disable thinking, which drops the effort.
    thinking = reasoning_effort not in ("none", "minimal")
    template_effort = (
        reasoning_effort
        if reasoning_effort in ("low", "medium", "high", "xhigh")
        else None
    )
    # Render the prompt
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        reasoning_effort=template_effort,
    )
    # The template has no thinking switch; per its own instructions a
    # non-thinking reply begins directly after the `</mm:think>` prefix.
    if not thinking:
        text = text + THINK_CLOSE
    # Tokenize
    return tokenizer.encode(text, add_special_tokens=False)

@compile(
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "transformers==5.15.0", "tokenizers==0.22.2")
        .pip_install("sglang==0.5.21", flags="--no-deps") # parsers only, skip CUDA dependency tree
        .pip_install(
            "aiohttp", "IPython", "msgspec", "openai", "orjson",
            "partial_json_parser", "pybase64", "pyzmq", "starlette", "xgrammar"
        )
        .env({
            "HF_TOKEN": environ.get("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_100",  # Compile for Blackwell
            tensor_parallelism=2,           # Run on 2xB200
            max_running_requests=64,
            max_total_tokens=MAX_TOTAL_TOKENS
        ),
        KVRoutingMetadata(tokenize=_tokenize)
    ]
)
def minimax_m3_1_flash(
    messages: Annotated[list[Message], Annotations.ChatMessages(
        description="Messages comprising the conversation so far.",
        context_length=MAX_TOTAL_TOKENS,
        batch=BatchConfig(mode="continuous")
    )],
    *,
    tools: Annotated[
        list[dict],
        Annotations.ChatTools(description="Tools the model may call.")
    ]=None,
    reasoning_effort: Annotated[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"],
        Annotations.ReasoningEffort(description="Model reasoning effort.")
    ]=None,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=262_144
    )]=32768,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.",
        min=0.0,
        max=2.0
    )]=1.0,
    top_p: Annotated[float, Annotations.TopP(
        description="Nucleus sampling probability.",
        min=0.0,
        max=1.0
    )]=0.95,
    seed: Annotated[
        int,
        Annotations.Seed(description="Seed for reproducible sampling.")
    ]=None,
) -> Iterator[ChatCompletionChunk]:
    """
    Stream chat completions from MiniMax-M3.1 Flash.
    """
    # Submit the request to the shared batching manager
    input_ids = _tokenize(messages, tools, reasoning_effort)
    # Check input token count
    if len(input_ids) >= MAX_TOTAL_TOKENS:
        raise ValueError(
            f"The input ({len(input_ids)} tokens) is longer than the model's "
            f"context length ({MAX_TOTAL_TOKENS} tokens)."
        )
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    prompt_tokens = len(input_ids)
    manager.add_request(
        input_ids=input_ids,
        request_id=completion_id,
        streaming=True,
        max_new_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    # Stream the completion
    try:
        # Create parsers for reasoning and function calling.
        # Thinking is adaptive, so reasoning is never forced: the parser
        # enters it only when the model opens `<mm:think>`, and drops the
        # stray leading `</mm:think>` a non-thinking reply may open with.
        decoder = DecodeStream(skip_special_tokens=False)
        reasoning = ReasoningParser(
            "minimax-m3",
            stream_reasoning=True,
            force_reasoning=False
        )
        functions = FunctionCallParser(_tool_specs(tools), "minimax-m3")
        # Initialize decode state
        completion_tokens = 0
        reasoning_tokens = 0
        pending_tokens = 0
        tool_calls = 0
        tool_call_id = ""
        held_newlines = ""
        trim_content = False
        error = ""
        role_sent = False
        # Consume generated tokens
        for output in manager.request_id_iter(request_id=completion_id):
            if output.status == RequestStatus.FAILED:
                error = output.error or "The request was aborted."
                break
            if not role_sent:
                # First chunk announces the assistant role with no content (OpenAI protocol)
                yield _chunk(completion_id, created, DeltaMessage(role="assistant", content=""))
                role_sent = True
            finished = output.status == RequestStatus.FINISHED
            # Decode the new tokens and split the text into reasoning and normal text
            segments = []
            text = ""
            new_tokens = output.generated_tokens[completion_tokens:]
            # Trim the matched stop token before decoding
            if finished and new_tokens and new_tokens[-1] in STOP_TOKENS:
                new_tokens = new_tokens[:-1]
            for token_id in new_tokens:
                pending_tokens += 1
                piece = decoder.step(tokenizer.backend_tokenizer, token_id)
                if piece is not None:
                    text += piece
            if text:
                reasoning_text, normal_text = reasoning.parse_stream_chunk(text)
                segments.append((reasoning_text, normal_text, pending_tokens))
                pending_tokens = 0
            completion_tokens = len(output.generated_tokens)
            if finished:
                reasoning_text, normal_text = reasoning.parse_stream_end()
                segments.append((reasoning_text, normal_text, 0))
            # Render the segments as OpenAI chunks
            for index, segment in enumerate(segments):
                reasoning_text, normal_text, span = segment
                if reasoning_text:
                    reasoning_tokens += span
                    text = held_newlines + reasoning_text
                    kept = text.rstrip("\n")
                    held_newlines = text[len(kept):]
                    trim_content = True
                    if kept:
                        yield _chunk(completion_id, created, DeltaMessage(reasoning_content=kept))
                if not normal_text and not finished:
                    continue
                calls = []
                if tools:
                    normal_text, calls = functions.parse_stream_chunk(normal_text)
                    if finished and index == len(segments) - 1:
                        tail, more = functions.parse_stream_end()
                        normal_text = normal_text + tail
                        calls = calls + more
                for call in calls:
                    if call.name:
                        tool_call_id = f"call_{uuid4()}"
                        tool_calls += 1
                    tool_call = ChoiceDeltaToolCall(
                        index=tool_calls - 1,
                        id=tool_call_id if call.name else None,
                        type="function" if call.name else None,
                        function=ChoiceDeltaToolCall.Function(
                            name=call.name,
                            arguments=call.parameters
                        )
                    )
                    yield _chunk(completion_id, created, DeltaMessage(tool_calls=[tool_call]))
                if normal_text:
                    held_newlines = ""
                    if trim_content:
                        normal_text = normal_text.lstrip("\n")
                        if normal_text:
                            trim_content = False
                    if normal_text:
                        yield _chunk(completion_id, created, DeltaMessage(content=normal_text))
            if finished:
                finish_reason = _finish_reason(
                    completion_tokens=completion_tokens,
                    max_output_tokens=max_output_tokens,
                    tool_calls=tool_calls
                )
                usage = ChatCompletion.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                        cached_tokens=getattr(output, "cached_tokens", 0),
                    ),
                    completion_tokens_details=ChatCompletion.Usage.CompletionTokensDetails(
                        reasoning_tokens=reasoning_tokens,
                    ),
                )
                yield _chunk(completion_id, created, DeltaMessage(content=""), finish_reason, usage)
        if error:
            raise RuntimeError(error)
    finally:
        # Release the engine request so it stops holding KV and a batch slot.
        # Cancelling a request that already finished is a no-op.
        manager.cancel_request(request_id=completion_id)

def _tool_specs(tools: list[dict] | None) -> list[Tool]:
    """
    Convert OpenAI tool definitions into SGLang tool specs.
    """
    if not tools:
        return []
    return [Tool(type=tool["type"], function=tool["function"]) for tool in tools]

def _chunk(
    completion_id: str,
    created: int,
    delta: DeltaMessage,
    finish_reason: str | None = None,
    usage: ChatCompletion.Usage | None = None
) -> ChatCompletionChunk:
    """
    Construct a single-choice streaming chunk.
    """
    return ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=CHECKPOINT,
        choices=[StreamChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason,
        )],
        usage=usage,
    )

def _finish_reason(
    completion_tokens: int,
    max_output_tokens: int,
    tool_calls: int
) -> str:
    """
    Compute the finish reason for a completed generation.
    """
    if completion_tokens >= max_output_tokens: return "length"
    if tool_calls > 0: return "tool_calls"
    return "stop"
