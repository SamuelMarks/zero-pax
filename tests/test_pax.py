"""Tests for zero_pax API."""

from zero_pax import BaseParameter, Layer


def test_base_parameter():
    """Executes the test_base_parameter test.

    Returns:
        The result of the test.
    """
    p = BaseParameter(1.0)
    assert p.value == 1.0


def test_layer():
    """Executes the test_layer test.

    Returns:
        The result of the test.
    """
    layer_obj = Layer()
    layer_obj.register_parameter("w", 2.0)
    assert "w" in layer_obj.params
    assert layer_obj.params["w"].value == 2.0
    assert layer_obj(1) == 1


def test_base_parameter_call():
    """Executes the test_base_parameter_call test.

    Returns:
        The result of the test.
    """
    from zero_pax import BaseParameter

    p = BaseParameter(42.0)
    assert p() == 42.0
