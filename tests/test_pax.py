"""Tests for zero_pax API."""

from zero_pax import BaseParameter, Layer


def test_base_parameter():
    """Test BaseParameter."""
    p = BaseParameter(1.0)
    assert p.value == 1.0


def test_layer():
    """Test Layer."""
    layer_obj = Layer()
    layer_obj.register_parameter("w", 2.0)
    assert "w" in layer_obj.params
    assert layer_obj.params["w"].value == 2.0
    assert layer_obj(1) == 1
