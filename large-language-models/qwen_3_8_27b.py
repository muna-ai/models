#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["accelerate", "muna", "torch", "transformers>=5.12"]
# ///

from __future__ import annotations
from accelerate import init_empty_weights
from enum import IntEnum
from json import dumps
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import (
    Annotations, KVRoutingMetadata, SpeculativeDecodingConfig,
    TorchToSGLangInferenceMetadata
)
from muna.beta.openai import (
    ChatCompletion, ChatCompletionChunk, ChoiceDeltaToolCall,
    DeltaMessage, Message, StreamChoice
)
from os import environ
from pydantic import BaseModel
from time import time
from torch.nn import Module
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM
from transformers.generation import ContinuousBatchingConfig, GenerationConfig
from transformers.generation.continuous_batching import RequestStatus
from typing import Annotated, Iterator
from uuid import uuid4

# Load the Qwen 3.8 text backbone
CHECKPOINT = "RadixArk/Qwen3.8-27B-NVFP4"
config = AutoConfig.from_pretrained(CHECKPOINT)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
text_config = config.text_config
text_config._name_or_path = CHECKPOINT
with init_empty_weights():
    model = Qwen3_5ForCausalLM._from_config(
        text_config,
        attn_implementation="eager",
    )

# Load the DFlash2 draft model config.
# Transformers has no `DFlash2DraftModel` class and the checkpoint ships no
# remote code, so the drafter cannot be instantiated here. The compiler only
# needs a module carrying the config.
DRAFT_CHECKPOINT = "z-lab/Qwen3.8-27B-DFlash2"
draft_config = AutoConfig.from_pretrained(DRAFT_CHECKPOINT)

class DFlash2DraftStub(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

draft_model = DFlash2DraftStub(draft_config)

# Resolve the reasoning and tool call marker tokens
THINK_OPEN = tokenizer.convert_tokens_to_ids("<think>")
THINK_CLOSE = tokenizer.convert_tokens_to_ids("</think>")
TOOL_OPEN = tokenizer.convert_tokens_to_ids("<tool_call>")
TOOL_CLOSE = tokenizer.convert_tokens_to_ids("</tool_call>")

# Response template for one tool call body. Qwen 3.8 emits tool calls as
# `<function=NAME><parameter=KEY>VALUE</parameter>...</function>` inside
# `<tool_call>` markers.
TOOL_CALL_TEMPLATE = {
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "tool_calls": {
            "open_pattern": r"<function=(?P<name>[\w.\-]+)>",
            "close": "</function>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": r"<parameter=(?P<key>[\w.\-]+)>\n?(?P<value>.*?)\n?</parameter>"
            },
            "transform": {
                "type": "function",
                "function": { "name": "{name}", "arguments": "{content}" }
            }
        }
    }
}

# Create the continuous batching manager
generation_config = GenerationConfig(
    max_new_tokens=2048,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    do_sample=True,
    temperature=0.7,
    top_k=50,
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
# This function is both used for both inference and KV-aware routing
def _tokenize(
    messages: list[Message],
    tools: list[dict] | None = None
) -> list[int]:
    return tokenizer.apply_chat_template(
        [{
            "role": m.role,
            "content": m.content,
            "tool_calls": m.tool_calls,
            "tool_call_id": m.tool_call_id
        } for m in messages],
        tools=tools,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )

@compile(
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "transformers>=5.12")
        .env({
            "HF_TOKEN": environ.get("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_100",  # Compile for Blackwell
            tensor_parallelism=1,
            speculative_decoding=SpeculativeDecodingConfig(
                draft_model=draft_model,
                num_draft_tokens=8,         # DFlash2 block size
            ),
            max_running_requests=4,
            max_total_tokens=32_768
        ),
        KVRoutingMetadata(tokenize=_tokenize)
    ]
)
def qwen_3_8_27b(
    messages: Annotated[list[Message], Parameter.Generic(
        description="Messages comprising the conversation so far.",
        batch=BatchConfig(mode="continuous")
    )],
    *,
    tools: Annotated[
        list[dict],
        Annotations.ChatTools(description="Tools the model may call."
    )]=None,
    max_output_tokens: Annotated[int, Annotations.MaxOutputTokens(
        description="Maximum number of tokens in the response.",
        min=1,
        max=32768
    )]=32768,
    temperature: Annotated[float, Annotations.Temperature(
        description="Sampling temperature.",
        min=0.0,
        max=2.0
    )]=0.7,
) -> Iterator[ChatCompletionChunk]:
    """
    Stream chat completions from Qwen 3.8 27B (NVFP4).
    """
    # Submit the request to the shared batching manager. Other concurrent calls
    # to this predictor add their own requests in parallel; the manager merges
    # them all into the next forward step.
    input_ids = _tokenize(messages, tools)
    completion_id = f"chatcmpl-{uuid4()}"
    created = int(time())
    prompt_tokens = len(input_ids)
    manager.add_request(
        input_ids=input_ids,
        request_id=completion_id,
        streaming=True,
        max_new_tokens=max_output_tokens,
        temperature=temperature
    )
    # First chunk announces the assistant role with no content (match OpenAI protocol)
    yield _chunk(completion_id, created, DeltaMessage(role="assistant", content=""))
    # Compose the token pipeline: raw tokens, reasoning, and tools
    events = _create_token_stream(completion_id)
    events = _split_token_stream(
        events,
        open_id=THINK_OPEN,
        close_id=THINK_CLOSE,
        out_kind=_EventKind.REASONING,
        buffered=False,
        initial=_starts_in_reasoning(input_ids)
    )
    events = _split_token_stream(
        events,
        open_id=TOOL_OPEN,
        close_id=TOOL_CLOSE,
        out_kind=_EventKind.TOOL_CALL,
        buffered=True,
        initial=False
    )
    # Render events as OpenAI chunks
    reasoning_tokens = 0
    tool_calls = 0
    for event in events:
        match event.kind:
            case _EventKind.REASONING:
                reasoning_tokens += len(event.token_ids)
                text = tokenizer.decode(event.token_ids, skip_special_tokens=True)
                if text:
                    yield _chunk(completion_id, created, DeltaMessage(reasoning_content=text))
            case _EventKind.TOKENS:
                text = tokenizer.decode(event.token_ids, skip_special_tokens=True)
                if text:
                    yield _chunk(completion_id, created, DeltaMessage(content=text))
            case _EventKind.TOOL_CALL:
                text = tokenizer.decode(event.token_ids, skip_special_tokens=True)
                message = tokenizer.parse_response(
                    text,
                    TOOL_CALL_TEMPLATE,
                    prefix="",
                    tools=tools
                )
                function = message["tool_calls"][0]["function"]
                tool_call = ChoiceDeltaToolCall(
                    index=tool_calls,
                    id=f"call_{uuid4()}",
                    type="function",
                    function=ChoiceDeltaToolCall.Function(
                        name=function["name"],
                        arguments=dumps(function["arguments"])
                    )
                )
                tool_calls += 1
                yield _chunk(completion_id, created, DeltaMessage(tool_calls=[tool_call]))
            case _EventKind.FINISHED:
                finish_reason = _finish_reason(
                    completion_tokens=event.completion_tokens,
                    max_output_tokens=max_output_tokens,
                    tool_calls=tool_calls
                )
                usage = ChatCompletion.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=event.completion_tokens,
                    total_tokens=prompt_tokens + event.completion_tokens,
                    prompt_tokens_details=ChatCompletion.Usage.PromptTokensDetails(
                        cached_tokens=event.cached_tokens,
                    ),
                    completion_tokens_details=ChatCompletion.Usage.CompletionTokensDetails(
                        reasoning_tokens=reasoning_tokens,
                    ),
                )
                yield _chunk(completion_id, created, DeltaMessage(content=""), finish_reason, usage)

def _create_token_stream(request_id: str) -> Iterator[_Event]:
    """
    Transform the batching manager's chunk stream into a token stream.
    """
    seen = 0
    for chunk in manager.request_id_iter(request_id=request_id):
        new_token_ids = chunk.generated_tokens[seen:]
        seen = len(chunk.generated_tokens)
        if new_token_ids:
            yield _Event(kind=_EventKind.TOKENS, token_ids=new_token_ids)
        if chunk.status == RequestStatus.FINISHED:
            yield _Event(
                kind=_EventKind.FINISHED,
                completion_tokens=seen,
                cached_tokens=getattr(chunk, "cached_tokens", 0)
            )
            return

def _split_token_stream(
    upstream: Iterator[_Event],
    open_id: int,
    close_id: int,
    out_kind: int,
    buffered: bool,
    initial: bool
) -> Iterator[_Event]:
    """
    Map token spans between `open_id` and `close_id` and relabel them as `out_kind`.
    All other tokens are passthrough, while marker tokens are consumed.
    When `buffered`, held tokens are emitted as one event at the close marker instead of streamed.
    """
    inside = initial
    buffer = [0][:0]
    for event in upstream:
        if event.kind == _EventKind.FINISHED and inside and buffered and buffer:
            yield _Event(kind=out_kind, token_ids=buffer)
            yield event
            continue
        if event.kind != _EventKind.TOKENS:
            yield event
            continue
        ids = event.token_ids
        while ids:
            marker = close_id if inside else open_id
            crossed = marker in ids
            boundary = ids.index(marker) if crossed else len(ids)
            span = ids[:boundary]
            ids = ids[boundary + 1:]
            if not inside:
                if span:
                    yield _Event(kind=_EventKind.TOKENS, token_ids=span)
            elif buffered:
                buffer = buffer + span
                if crossed:
                    yield _Event(kind=out_kind, token_ids=buffer)
                    buffer.clear()
            else:
                if span:
                    yield _Event(kind=out_kind, token_ids=span)
            if crossed:
                inside = not inside

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

def _starts_in_reasoning(input_ids: list[int]) -> bool:
    """
    The Qwen3.8 template ends the generation prompt with `<think>\n` when
    thinking is enabled (trailing newline, so the *last* token is not the
    marker) and with a closed `<think>\n\n</think>\n\n` block when disabled.
    Scan back to the most recent marker instead of checking the final token.
    """
    for token in reversed(input_ids):
        if token == THINK_OPEN:
            return True
        if token == THINK_CLOSE:
            return False
    return False

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

class _EventKind(IntEnum):
    """
    Kind of an event flowing through the token pipeline.
    """
    TOKENS = 0      # unclaimed tokens, destined for `content`
    REASONING = 1   # tokens inside `<think>`...`</think>`
    TOOL_CALL = 2   # inner tokens of one complete `<tool_call>` block
    FINISHED = 3    # terminal event, carries stream totals

class _Event(BaseModel):
    """
    One event in the token pipeline.
    """
    kind: int
    token_ids: list[int] = []
    completion_tokens: int = 0  # FINISHED only: total generated, markers included
    cached_tokens: int = 0      # FINISHED only: prefix-cache hits