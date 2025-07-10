from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Any, Union

BackendParameters = dict[str, Any]
GateParameters = list[float]

PauliError = dict[
    str, float
]  # A dictionary with keys 'X', 'Y', 'Z' and their corresponding probabilities.

NoiseFunction = Callable[[GateParameters, BackendParameters], PauliError]
TimeFunction = Callable[[GateParameters, BackendParameters], float]


def _to_hashable(param: Union[GateParameters, BackendParameters]):
    """
    Generate a hash for the given parameters.

    Args:
        param (GateParameters | BackendParameters): The parameters to hash.
    """
    return (
        tuple(sorted(param.items()))
        if isinstance(param, dict)
        else tuple(sorted(param))
    )


@dataclass
class NoiseModel:
    """
    A model for computing noise probabilities for quantum gates based on gate
    parameters and backend parameters.

    Attributes:
        gate_name (str): The name of the gate for which the noise model is defined.
        noise_fn (Callable[[GateParameters, BackendParameters], PauliError]): A callable that computes
            the noise probabilities for the gate given its parameters and backend parameters.
            It should return a dictionary with keys 'X', 'Y', and 'Z' representing the
            probabilities of the respective Pauli errors.
        cache_size (int): The size of the cache for the noise function. Default is 128.
    """

    gate_name: str
    noise_fn: NoiseFunction
    cache_size: int = 128  # Default cache size for the noise function

    def __post_init__(self):
        """
        After initialization, we create a cached version of the noise function
        """

        @lru_cache(maxsize=self.cache_size)
        def _cached_noise_fn(
            gate_params_tuple: tuple[float, ...],
            backend_params_tuple: tuple[tuple[str, Any], ...],
        ) -> PauliError:
            """Cached version of the noise function to improve performance."""
            gate_params = list(gate_params_tuple)
            backend_params = dict(backend_params_tuple)
            return self.noise_fn(gate_params, backend_params)

        self._cached_noise_fn = _cached_noise_fn

    def _get_cached_noise_fn(
        self, gate_params: GateParameters, backend_params: BackendParameters
    ) -> PauliError:
        """
        Get the cached noise function for the given gate and backend parameters.

        Args:
            gate_params (GateParameters): Parameters specific to the gate.
            backend_params (BackendParameters): Parameters specific to the backend.

        Returns:
            PauliError: The computed noise probabilities.
        """
        gate_params_tuple = _to_hashable(gate_params)
        backend_params_tuple = _to_hashable(backend_params)
        return self._cached_noise_fn(gate_params_tuple, backend_params_tuple)

    def _validate(
        self, gate_params: GateParameters, backend_params: BackendParameters
    ) -> PauliError:
        """Validate the noise model for the given backend parameters and instruction."""
        if not callable(self.apply):
            raise ValueError('Noise functions must be callable.')

        try:
            error = self._get_cached_noise_fn(gate_params, backend_params)
        except KeyError as e:
            raise ValueError(
                f"Missing required parameter for gate '{self.gate_name}': {e}"
            )
        except TypeError as e:
            raise ValueError(
                f"Invalid parameter type for gate '{self.gate_name}': {e}"
            )
        except Exception as e:
            raise ValueError(
                f"Error computing noise for gate '{self.gate_name}': {e}"
            )

        x_prob = error.get('X', 0.0)
        y_prob = error.get('Y', 0.0)
        z_prob = error.get('Z', 0.0)

        if not (0 <= x_prob <= 1 and 0 <= y_prob <= 1 and 0 <= z_prob <= 1):
            raise ValueError(
                'Noise probabilities must be in the range [0, 1].'
            )

        total_prob = x_prob + y_prob + z_prob

        if not 0 <= total_prob <= 1:
            raise ValueError(
                'Total noise probability must be in the range [0, 1].'
            )

        return error

    def apply(
        self, gate_params: GateParameters, backend_params: BackendParameters
    ) -> PauliError:
        """
        Apply the noise model to compute the noise probabilities for a gate.
        It validates the noise probabilities and returns them.

        Raises:
            ValueError: If the noise probabilities are not valid.

        Args:
            gate_params (GateParameters): Parameters specific to the gate.
            backend_params (BackendParameters): Parameters specific to the backend.
        """
        return self._validate(gate_params, backend_params)

    def clear_cache(self):
        """
        Clear the cache of the noise function.
        """
        self._cached_noise_fn.cache_clear()

    def cache_info(self) -> dict[str, Any]:
        """
        Get the cache information of the noise function.

        Returns:
            dict[str, Any]: Cache statistics including hits, misses, and size.
        """
        return self._cached_noise_fn.cache_info()


@dataclass
class TimeModel:
    """
    A model for computing the duration of quantum gates based on gate parameters
    and backend parameters.

    Attributes:
        gate_name (str): The name of the gate for which the time model is defined.
        time_fn (TimeFunction): A callable that computes the duration of a gate
            given its parameters and backend parameters.
    """

    gate_name: str
    time_fn: TimeFunction
    cache_size: int = 128  # Default cache size for the time function

    def __post_init__(self):
        """Initialize the LRU cache after dataclass initialization."""

        @lru_cache(maxsize=self.cache_size)
        def _cached_time_fn(gate_params_tuple, backend_params_tuple):
            gate_params = list(gate_params_tuple)
            backend_params = dict(backend_params_tuple)
            return self.time_fn(gate_params, backend_params)

        self._cached_time_fn = _cached_time_fn

    def _get_cached_time_fn(
        self, gate_params: GateParameters, backend_params: BackendParameters
    ) -> float:
        """
        Get the cached time function for the given gate and backend parameters.
        """
        gate_params_tuple = _to_hashable(gate_params)
        backend_params_tuple = _to_hashable(backend_params)
        return self._cached_time_fn(gate_params_tuple, backend_params_tuple)

    def _validate(
        self, gate_params: GateParameters, backend_params: BackendParameters
    ) -> float:
        """
        Validate the time model for the given backend parameters and instruction,
        and compute the duration of the gate.
        """
        if not callable(self.apply):
            raise ValueError('Time function must be callable.')

        duration = self._get_cached_time_fn(gate_params, backend_params)

        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError('Duration must be a non-negative number.')

        return duration

    def apply(
        self, gate_params: GateParameters, backend_params: BackendParameters
    ) -> float:
        """
        Apply the time model to compute the duration of a gate.
        It validates the duration and returns it.

        Raises:
            ValueError: If the duration is not valid.

        Args:
            gate_params (GateParameters): Parameters specific to the gate.
            backend_params (BackendParameters): Parameters specific to the backend.
        """
        return self._validate(gate_params, backend_params)

    def clear_cache(self):
        """Clear the LRU cache."""
        self._cached_time_fn.cache_clear()

    def cache_info(self):
        """Return cache statistics."""
        return self._cached_time_fn.cache_info()
