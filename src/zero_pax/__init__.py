"""Core functionality for the __init__ module."""

import ml_switcheroo

"""Public API definitions for zero_pax."""


class BaseParameter:
    """Represents the BaseParameter configuration and behavior.

    This class encapsulates the functionality for BaseParameter.
    """

    def __init__(self, value):
        """Initializes the object.

        Args:
            value: The value parameter.

        Returns:
            The result of the operation.
        """
        self.value = value

    def __call__(self):
        """Calls the object as a function.

        Returns:
            The result of the operation.
        """
        return self.value


class Layer:
    """Represents the Layer configuration and behavior.

    This class encapsulates the functionality for Layer.
    """

    def __init__(self, name=None):
        """Initializes the object.

        Args:
            name: The name parameter.

        Returns:
            The result of the operation.
        """
        self.name = name
        self.params = {}

    def register_parameter(self, name, value):
        """register_parameter function.

        Args:
            name: The name parameter.
            value: The value parameter.

        Returns:
            The result of the operation.
        """
        self.params[name] = BaseParameter(value)

    def __call__(self, inputs):
        """Calls the object as a function.

        Args:
            inputs: The inputs parameter.

        Returns:
            The result of the operation.
        """
        return inputs
