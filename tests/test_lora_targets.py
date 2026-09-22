"""
LoRA target-module selection, which failed silently once and cost a run.

The top-30 adapter was trained with a config naming the seven Llama
projections against a phi-3 base. phi-3 fuses q/k/v into qkv_proj and gate/up
into gate_up_proj, so five of the seven matched nothing. PEFT adapted what it
recognised, said nothing about the rest, and wrote all seven names into
adapter_config.json anyway -- so the config on disk disagreed with the
weights, and no log line, loss value or checkpoint revealed it.

These tests use small synthetic modules rather than real checkpoints, so they
need no network and no GPU.
"""

import pytest
import torch
from torch import nn

from litigpt.training.trainer import (
    discover_target_modules,
    validate_target_modules,
)

LLAMA_NAMES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
PHI3_NAMES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]


def _block(projection_names):
    """A stand-in transformer block exposing the given projections."""
    block = nn.Module()
    for name in projection_names:
        setattr(block, name, nn.Linear(8, 8))
    return block


def _model(projection_names, n_layers=2, with_head=True):
    model = nn.Module()
    model.layers = nn.ModuleList(
        [_block(projection_names) for _ in range(n_layers)]
    )
    if with_head:
        model.lm_head = nn.Linear(8, 32)
    return model


class TestDiscovery:
    def test_finds_llama_projections(self):
        assert discover_target_modules(_model(LLAMA_NAMES)) == sorted(LLAMA_NAMES)

    def test_finds_phi3_fused_projections(self):
        assert discover_target_modules(_model(PHI3_NAMES)) == sorted(PHI3_NAMES)

    def test_excludes_the_lm_head(self):
        # Adapting the head is a separate decision with a large parameter
        # cost; auto-detection must not opt you into it.
        assert "lm_head" not in discover_target_modules(_model(PHI3_NAMES))

    def test_deduplicates_across_layers(self):
        # Ten layers, still four names.
        assert discover_target_modules(
            _model(PHI3_NAMES, n_layers=10)
        ) == sorted(PHI3_NAMES)

    def test_raises_when_there_is_nothing_to_adapt(self):
        empty = nn.Module()
        empty.embed = nn.Embedding(4, 4)
        with pytest.raises(RuntimeError, match="no linear projections"):
            discover_target_modules(empty)

    def test_matches_quantized_linear_classes(self):
        """
        Under 4-bit loading bitsandbytes replaces nn.Linear with Linear4bit.
        An isinstance check against nn.Linear would find nothing in exactly
        the configuration this project trains in, so detection matches on
        class name instead.
        """
        class Linear4bit(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 2))

        model = nn.Module()
        block = nn.Module()
        block.qkv_proj = Linear4bit()
        block.down_proj = Linear4bit()
        model.layers = nn.ModuleList([block])
        assert discover_target_modules(model) == ["down_proj", "qkv_proj"]


class TestValidation:
    def test_accepts_names_that_exist(self):
        validate_target_modules(_model(PHI3_NAMES), PHI3_NAMES)

    def test_accepts_a_deliberate_subset(self):
        # Adapting only attention is a legitimate choice.
        validate_target_modules(_model(LLAMA_NAMES), ["q_proj", "v_proj"])

    def test_rejects_the_llama_list_against_phi3(self):
        """The exact mistake that produced the first top-30 adapter."""
        with pytest.raises(ValueError) as exc:
            validate_target_modules(_model(PHI3_NAMES), LLAMA_NAMES)

        message = str(exc.value)
        # The error has to be actionable: what was missing, and what to use.
        for missing in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]:
            assert missing in message
        assert "gate_up_proj" in message and "qkv_proj" in message

    def test_rejects_even_one_bad_name(self):
        # Partial adaptation is the dangerous case, not a tolerable one.
        with pytest.raises(ValueError):
            validate_target_modules(_model(PHI3_NAMES), PHI3_NAMES + ["q_proj"])

    def test_llama_list_still_valid_for_a_llama_model(self):
        validate_target_modules(_model(LLAMA_NAMES), LLAMA_NAMES)
