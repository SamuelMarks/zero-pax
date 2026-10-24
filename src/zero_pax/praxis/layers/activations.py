"""Activation layers for zero_pax.praxis.layers."""

from __future__ import annotations

from zero_pax import _tensor_ops as np
from zero_pax.praxis.base import BasePraxisLayer
from zero_pax.types import Tensor, TensorOrScalar


class BaseActivation(BasePraxisLayer):
    """Base class for all elementwise activation layers."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Apply the base identity activation to `inputs`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Activated `Tensor`.
        """
        return np.asarray(inputs)


class CubedReLU(BaseActivation):
    """Cubed Rectified Linear Unit activation layer: `max(0, x)^3`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `max(0, inputs) ** 3`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Cubed ReLU output `Tensor`.
        """
        x = np.asarray(inputs)
        return np.maximum(0.0, x) ** 3


class ELU(BaseActivation):
    """Exponential Linear Unit activation layer.

    Attributes:
        alpha: Scale factor for the negative factor (`alpha * (exp(x) - 1)`).
    """

    alpha: float = 1.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `where(inputs > 0, inputs, alpha * (exp(inputs) - 1))`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ELU activated `Tensor`.
        """
        x = np.asarray(inputs)
        return np.where(x > 0.0, x, self.alpha * (np.exp(x) - 1.0))


class GELU(BaseActivation):
    """Gaussian Error Linear Unit activation layer.

    Attributes:
        approximate: Whether to use the tanh approximation (`True`) or exact erf (`False`).
    """

    approximate: bool = True

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute the GELU activation of `inputs`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            GELU activated `Tensor`.
        """
        x = np.asarray(inputs)
        if self.approximate:
            cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))
            return x * cdf
        return 0.5 * x * (1.0 + np.erf(x / np.sqrt(2.0)))


class Identity(BaseActivation):
    """Identity activation layer that returns inputs unchanged."""


class LeakyReLU(BaseActivation):
    """Leaky Rectified Linear Unit activation layer.

    Attributes:
        negative_slope: Slope multiplier for negative inputs.
    """

    negative_slope: float = 0.0

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `where(inputs > 0, inputs, negative_slope * inputs)`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            LeakyReLU activated `Tensor`.
        """
        x = np.asarray(inputs)
        return np.where(x > 0.0, x, self.negative_slope * x)


class ReLU(BaseActivation):
    """Rectified Linear Unit activation layer: `max(0, x)`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `max(0, inputs)`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ReLU activated `Tensor`.
        """
        return np.maximum(0.0, np.asarray(inputs))


class ReLU6(BaseActivation):
    """Rectified Linear Unit capped at 6: `min(max(0, x), 6)`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `min(max(0, inputs), 6)`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            ReLU6 activated `Tensor`.
        """
        return np.minimum(np.maximum(0.0, np.asarray(inputs)), 6.0)


class Sigmoid(BaseActivation):
    """Logistic sigmoid activation layer: `1 / (1 + exp(-x))`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `1 / (1 + exp(-inputs))`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Sigmoid activated `Tensor`.
        """
        x = np.asarray(inputs)
        return 1.0 / (1.0 + np.exp(-x))


class Swish(BaseActivation):
    """Swish activation layer: `x * sigmoid(x)`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `inputs * sigmoid(inputs)`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Swish activated `Tensor`.
        """
        x = np.asarray(inputs)
        return x / (1.0 + np.exp(-x))


class SiLU(Swish):
    """Sigmoid Linear Unit (Swish) activation layer: `x / (1 + exp(-x))`."""


class SquaredReLU(BaseActivation):
    """Squared Rectified Linear Unit activation layer: `max(0, x)^2`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `square(max(0, inputs))`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            SquaredReLU activated `Tensor`.
        """
        return np.square(np.maximum(0.0, np.asarray(inputs)))


class Tanh(BaseActivation):
    """Hyperbolic tangent activation layer: `tanh(x)`."""

    def __call__(
        self,
        inputs: TensorOrScalar,
        *args: TensorOrScalar,
        **kwargs: TensorOrScalar,
    ) -> Tensor:
        """Compute `tanh(inputs)`.

        Args:
            inputs: Input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tanh activated `Tensor`.
        """
        return np.tanh(np.asarray(inputs))


__all__ = [
    "BaseActivation",
    "CubedReLU",
    "ELU",
    "GELU",
    "Identity",
    "LeakyReLU",
    "ReLU",
    "ReLU6",
    "SiLU",
    "Sigmoid",
    "SquaredReLU",
    "Swish",
    "Tanh",
]
