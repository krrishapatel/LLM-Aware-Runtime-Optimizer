"""Small models to test against.

Nothing here downloads anything. The whole suite runs on hand-built modules, so
CI installs torch and pytest and finishes in seconds instead of pulling a
HuggingFace checkpoint.
"""

import pytest
import torch
import torch.nn as nn
import torch.ao.quantization as tq


class TinyTransformerBlock(nn.Module):
    """One attention block plus a feedforward, with the layer names a real
    transformer uses so the name-based attention count has something to find."""

    def __init__(self, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.attention_query = nn.Linear(hidden, hidden)
        self.attention_key = nn.Linear(hidden, hidden)
        self.attention_value = nn.Linear(hidden, hidden)
        self.attention_output = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.feedforward_in = nn.Linear(hidden, hidden * 2)
        self.feedforward_out = nn.Linear(hidden * 2, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.heads = heads

    def forward(self, x):
        q = self.attention_query(x)
        k = self.attention_key(x)
        v = self.attention_value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5, dim=-1)
        attended = self.attention_output(scores @ v)
        x = self.norm1(x + attended)
        hidden = torch.relu(self.feedforward_in(x))
        return self.norm2(x + self.feedforward_out(hidden))


class TinyLM(nn.Module):
    """Embedding plus one block plus a projection back to the vocabulary."""

    def __init__(self, vocab: int = 64, hidden: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.block = TinyTransformerBlock(hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return self.head(self.block(self.embedding(input_ids)))


class StubbedMLP(nn.Module):
    """Wrapped in QuantStub/DeQuantStub, which is what static quantization needs."""

    def __init__(self, size: int = 64):
        super().__init__()
        self.quant = tq.QuantStub()
        self.fc1 = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(size, size)
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.dequant(x)


class EmbeddingHeavy(nn.Module):
    """Most of the weight is in an embedding, which dynamic quantization skips."""

    def __init__(self, vocab: int = 2000, hidden: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, 4)

    def forward(self, input_ids):
        return self.head(self.embedding(input_ids))


@pytest.fixture
def mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))


@pytest.fixture
def tiny_lm():
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture
def tiny_lm_input():
    torch.manual_seed(0)
    return torch.randint(0, 64, (2, 8), dtype=torch.long)


@pytest.fixture
def stubbed_mlp():
    torch.manual_seed(0)
    return StubbedMLP()


@pytest.fixture
def tiny_stubbed_mlp():
    """Small enough that int8 quantization makes the checkpoint bigger."""
    torch.manual_seed(0)
    return StubbedMLP(size=16)


@pytest.fixture
def embedding_heavy():
    torch.manual_seed(0)
    return EmbeddingHeavy()
