#
#   Muna
#   Copyright © 2026 NatML Inc. All Rights Reserved.
#

# /// script
# requires-python = ">=3.12"
# dependencies = ["accelerate", "muna", "sentence-transformers>=6", "torch"]
# ///

from __future__ import annotations
from accelerate import init_empty_weights
from contextlib import contextmanager
from muna import compile, BatchConfig, Parameter, Sandbox
from muna.beta import Annotations, TorchToSGLangInferenceMetadata
from muna.beta.openai import EmbeddingCreateResponse
from numpy import ndarray
from os import environ
from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules.transformer import Transformer
from transformers import AutoModel
from typing import Annotated

@contextmanager
def _empty_transformer_weights():
    """
    Skip downloading the weights during compilation.
    """
    def _load_model(
        self, model_name_or_path, transformer_task,
        config, backend, is_peft_model, **model_kwargs
    ):
        with init_empty_weights():
            return AutoModel.from_config(config)
    original = Transformer._load_model
    Transformer._load_model = _load_model
    try:
        yield
    finally:
        Transformer._load_model = original

# Load the Qwen3 Embedding model
with _empty_transformer_weights():
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="meta")

@compile(
    tag="@qwen/qwen-3-embedding-8b",
    access="public",
    targets=["x86_64-unknown-linux-gnu"],   # Linux x64 + CUDA only
    sandbox=Sandbox()
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .pip_install("accelerate", "sentence-transformers>=6")
        .env({
            "HF_TOKEN": environ.get("HF_TOKEN"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1"
        }),
    metadata=[
        TorchToSGLangInferenceMetadata(
            model=model,
            compute_architecture="sm_100",  # Compile for Blackwell
        )
    ]
)
def qwen_3_embedding_8b(
    texts: Annotated[list[str], Parameter.Generic(
        description="Input texts to embed.",
        batch=BatchConfig(mode="dynamic", capacity=64)
    )],
    *,
    instruct: Annotated[
        str,
        Parameter.Generic(description="Task instruction prepended to each text for query embeddings.")
    ]="",
    dimensions: Annotated[int, Annotations.EmbeddingDims(
        description="Embedding dimensions.",
        min=32,
        max=4096
    )]=4096
) -> tuple[
    Annotated[
        ndarray,
        Parameter.Embedding(description="Embedding matrix.")
    ],
    Annotated[
        EmbeddingCreateResponse.Usage,
        Parameter.Generic(description="Token usage.")
    ]
]:
    """
    Embed text using Qwen3 Embedding 8B.
    """
    # Format query instructions as `Instruct: {task}\nQuery:{text}`.
    prompt = f"Instruct: {instruct}\nQuery:" if instruct else ""
    # Check input length and raise exception instead of silently truncating
    features = model.preprocess(texts, prompt=prompt)
    lengths = features["attention_mask"].sum(dim=1)
    max_seq_length = model.max_seq_length
    if int(lengths.max()) >= max_seq_length:
        raise ValueError(
            f"Input exceeds the model's maximum context length of {max_seq_length} tokens. "
            f"Each text must be at most {max_seq_length - 1} tokens, including the instruction and special tokens."
        )
    # Embed
    embeddings = model.encode(
        texts,
        prompt=prompt,
        truncate_dim=dimensions,
        normalize_embeddings=True
    )
    # Create usage
    prompt_tokens = int(lengths.sum())
    usage = EmbeddingCreateResponse.Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens
    )
    # Return
    return embeddings, usage

if __name__ == "__main__":
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        "What is the capital of China?",
        "Explain gravity"
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the "
        "movement of planets around the sun.",
    ]
    # Embed queries (with instruction) and documents (without)
    query_embeddings, query_usage = qwen_3_embedding_8b(queries, instruct=task)
    doc_embeddings, doc_usage = qwen_3_embedding_8b(documents)
    scores = query_embeddings @ doc_embeddings.T
    print(f"Similarity scores:\n{scores}")
    print(f"Usage: queries={query_usage} documents={doc_usage}")
