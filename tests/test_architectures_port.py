"""Module docstring."""

import numpy as np
import pytest
from zero_pax.praxis.layers import (
    ResNet,
    ResNetBlock,
    VanillaNet,
    VanillaBlock,
    MLPBlock,
    Conformer,
    VitEntryLayers,
    VitExitLayers,
)


def test_resnet():
    """test_resnet docstring."""
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer = ResNetBlock()
    out = layer(inputs)
    assert out.shape == (2, 8, 8, 3)

    net = ResNet()
    out_net = net(inputs)
    assert out_net.shape == (2, 8, 8, 3)


def test_vanilla():
    """test_vanilla docstring."""
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer = VanillaBlock()
    out = layer(inputs)
    assert out.shape == (2, 8, 8, 3)

    net = VanillaNet()
    out_net = net(inputs)
    assert out_net.shape == (2, 8, 8, 3)


def test_mlpblock():
    """test_mlpblock docstring."""
    inputs = np.random.normal(size=(2, 10, 8)).astype(np.float32)
    layer = MLPBlock()
    out = layer(inputs)
    assert out.shape == (2, 10, 8)


def test_conformer():
    """test_conformer docstring."""
    inputs = np.random.normal(size=(2, 10, 8)).astype(np.float32)
    layer = Conformer()
    out = layer(inputs)
    assert out.shape == (2, 10, 8)


def test_vit_entry_exit():
    """test_vit_entry_exit docstring."""
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    entry = VitEntryLayers()
    out_entry = entry(inputs)
    # [B, H*W, C] -> 8*8 = 64
    assert out_entry.shape == (2, 64, 3)

    exit_layer = VitExitLayers()
    out_exit = exit_layer(out_entry)
    # [B, C]
    assert out_exit.shape == (2, 3)
