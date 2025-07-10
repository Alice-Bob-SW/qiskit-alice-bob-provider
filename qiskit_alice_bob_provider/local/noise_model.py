from dataclasses import dataclass
from typing import Callable, Any

BackendParameters = dict[str, Any]
GateParameters = list[float]

PauliError = dict[str, float] # A dictionary with keys 'X', 'Y', 'Z' and their corresponding probabilities.

NoiseFunction = Callable[[GateParameters, BackendParameters], PauliError]
TimeFunction = Callable[[GateParameters, BackendParameters], float]

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
    """
    gate_name: str
    noise_fn: NoiseFunction

    def _validate(self, gate_params: GateParameters, backend_params: BackendParameters) -> PauliError:
        """Validate the noise model for the given backend parameters and instruction."""
        if not callable(self.apply):
            raise ValueError('Noise functions must be callable.')

        try:
            error = self.noise_fn(gate_params, backend_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for gate '{self.gate_name}': {e}")
        except TypeError as e:
            raise ValueError(f"Invalid parameter type for gate '{self.gate_name}': {e}")
        except Exception as e:
            raise ValueError(f"Error computing noise for gate '{self.gate_name}': {e}")
        
        x_prob = error.get('X', 0.0)
        y_prob = error.get('Y', 0.0)
        z_prob = error.get('Z', 0.0)

        if not (0 <= x_prob <= 1 and 0 <= y_prob <= 1 and 0 <= z_prob <= 1):
            raise ValueError('Noise probabilities must be in the range [0, 1].')

        total_prob = x_prob + y_prob + z_prob

        if not 0 <= total_prob <= 1:
            raise ValueError('Total noise probability must be in the range [0, 1].')
        
        return error
    
    def apply(self, gate_params: GateParameters, backend_params: BackendParameters) -> PauliError:
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

    def _validate(self, gate_params: GateParameters, backend_params: BackendParameters) -> float:
        """
        Validate the time model for the given backend parameters and instruction,
        and compute the duration of the gate.
        """
        if not callable(self.apply):
            raise ValueError('Time function must be callable.')

        duration = self.time_fn(gate_params, backend_params)

        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError('Duration must be a non-negative number.')
        
        return duration
    
    def apply(self, gate_params: GateParameters, backend_params: BackendParameters) -> float:
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


