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
from typing import Annotated, Literal

@contextmanager
def _empty_transformer_weights():
    """
    Have `sentence_transformers.models.Transformer` construct its backbone from
    config on the meta device instead of downloading the safetensors shards.
    """
    def _load_model(
        self, model_name_or_path, transformer_task, config,
        backend, is_peft_model, **model_kwargs
    ):
        with init_empty_weights():
            return AutoModel.from_config(config)
    original = Transformer._load_model
    Transformer._load_model = _load_model
    try:
        yield
    finally:
        Transformer._load_model = original

# Load the Nomic Embed Text v1.5 model
with _empty_transformer_weights():
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", device="meta")
    max_seq_length = model.max_seq_length

@compile(
    tag="@nomic/nomic-embed-text-v1.5",
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
def nomic_embed_text_v1_5(
    texts: Annotated[list[str], Parameter.Generic(
        description="Input texts to embed.",
        batch=BatchConfig(mode="dynamic", capacity=64)
    )],
    *,
    task: Annotated[
        Literal["search_query", "search_document", "clustering", "classification"],
        Parameter.Generic(description="Task instruction prefix.")
    ]="search_document",
    dimensions: Annotated[int, Annotations.EmbeddingDims(
        description="Embedding dimensions.",
        min=64,
        max=768
    )]=768
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
    Embed text using Nomic Embed Text v1.5.
    """
    # Nomic requires a task instruction prefix on every text (`search_query: ...`).
    prompt = f"{task}: "
    # Check input length and raise exception instead of silently truncating
    features = model.preprocess(texts, prompt=prompt)
    lengths = features["attention_mask"].sum(dim=1)
    if int(lengths.max()) >= max_seq_length:
        raise ValueError(
            f"Input exceeds the model's maximum context length of {max_seq_length} tokens. "
            f"Each text must be at most {max_seq_length - 1} tokens, including the task prefix and special tokens."
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
    queries = [
        "What is TSNE?",
        "Who is Laurens van der Maaten?"
    ]
    documents = [
        "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
        "The capital of France is Paris."
    ]
    query_embeddings, query_usage = nomic_embed_text_v1_5(queries, task="search_query")
    doc_embeddings, doc_usage = nomic_embed_text_v1_5(documents, task="search_document")
    scores = query_embeddings @ doc_embeddings.T
    print(f"Similarity scores:\n{scores}")
    print(f"Usage: queries={query_usage} documents={doc_usage}")
