"""zero_pax API."""


class BaseParameter:
    """Base parameter class."""

    def __init__(self, value):
        """Initialize parameter.

        Args:
            value: The parameter value.
        """
        self.value = value


class Layer:
    """Base layer class."""

    def __init__(self):
        """Initialize layer."""
        self.params = {}

    def register_parameter(self, name, value):
        """Register a parameter.

        Args:
            name: The parameter name.
            value: The parameter value.
        """
        self.params[name] = BaseParameter(value)

    def __call__(self, inputs):
        """Forward pass.

        Args:
            inputs: The inputs.

        Returns:
            The inputs.
        """
        return inputs
