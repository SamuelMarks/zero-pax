"""Tests for the test_architectures_port module."""

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
    """Executes the test_resnet test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer = ResNetBlock()
    out = layer(inputs)
    assert out.shape == (2, 8, 8, 3)

    net = ResNet()
    out_net = net(inputs)
    assert out_net.shape == (2, 8, 8, 3)


def test_vanilla():
    """Executes the test_vanilla test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    layer = VanillaBlock()
    out = layer(inputs)
    assert out.shape == (2, 8, 8, 3)

    net = VanillaNet()
    out_net = net(inputs)
    assert out_net.shape == (2, 8, 8, 3)


def test_mlpblock():
    """Executes the test_mlpblock test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 8)).astype(np.float32)
    layer = MLPBlock()
    out = layer(inputs)
    assert out.shape == (2, 10, 8)


def test_conformer():
    """Executes the test_conformer test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 10, 8)).astype(np.float32)
    layer = Conformer()
    out = layer(inputs)
    assert out.shape == (2, 10, 8)


def test_vit_entry_exit():
    """Executes the test_vit_entry_exit test.

    Returns:
        The result of the test.
    """
    inputs = np.random.normal(size=(2, 8, 8, 3)).astype(np.float32)
    entry = VitEntryLayers()
    out_entry = entry(inputs)
    # [B, H*W, C] -> 8*8 = 64
    assert out_entry.shape == (2, 64, 3)

    exit_layer = VitExitLayers()
    out_exit = exit_layer(out_entry)
    # [B, C]
    assert out_exit.shape == (2, 3)
