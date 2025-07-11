from typing import Any, Callable, Iterator, List, Optional, Tuple

from qiskit_alice_bob_provider.local.noise_model import (
    NoiseFunction,
    NoiseModel,
    TimeFunction,
    TimeModel,
)
from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)
from qiskit_alice_bob_provider.processor.utils import pauli_errors_to_chi

_1Q_INSTRUCTIONS = [
    'x',
    'z',
    'p0',
    'p1',
    'p+',
    'p-',
    'mx',
    'mz',
    't',
    'tdg',
    'h',
    's',
    'sdg',
]


class CustomCat(ProcessorDescription):
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(
        self,
        backend_parameters: Optional[dict[str, Any]] = None,
        noise_models: Optional[dict[str, NoiseFunction]] = None,
        time_models: Optional[dict[str, TimeFunction]] = None,
        default_1q_noise_model: Optional[NoiseFunction] = None,
        default_1q_time_model: Optional[TimeFunction] = None,
        validate_parameters: Callable[
            [dict[str, float]], bool
        ] = lambda _: True,
        name: Optional[str] = 'CustomCat',
    ):
        self.backend_parameters = (
            backend_parameters if backend_parameters else {}
        )
        self.noise_models: dict[str, NoiseFunction] = (
            noise_models if noise_models else {}
        )
        self.time_models: dict[str, TimeFunction] = (
            time_models if time_models else {}
        )
        self.default_1q_noise_model: Optional[
            NoiseFunction
        ] = default_1q_noise_model
        self.default_1q_time_model: Optional[
            TimeFunction
        ] = default_1q_time_model
        self.validate_parameters: Callable[
            [dict[str, float]], bool
        ] = validate_parameters

        if not self.validate_parameters(self.backend_parameters):
            raise ValueError(f'Invalid parameters provided for {name}.')

        self.name = name
        _clock_cycle: float = self.backend_parameters.get('clock_cycle', 1e-9)
        _n_qubits: Optional[int] = self.backend_parameters.get(
            'n_qubits', None
        )

        if _n_qubits is None:
            raise ValueError("Backend parameters must include 'n_qubits'.")

        self.clock_cycle = _clock_cycle
        self.n_qubits = _n_qubits

        self._noise_model_instances: dict[str, NoiseModel] = {}
        self._time_model_instances: dict[str, TimeModel] = {}

    def _get_noise_model(self, gate_name: str) -> Optional[NoiseModel]:
        """Retrieve the noise model for a given gate name."""
        if gate_name in self._noise_model_instances:
            return self._noise_model_instances[gate_name]

        noise_model = None
        if gate_name in self.noise_models:
            noise_model = NoiseModel(
                gate_name=gate_name, noise_fn=self.noise_models[gate_name]
            )
        elif self.default_1q_noise_model and gate_name in _1Q_INSTRUCTIONS:
            noise_model = NoiseModel(
                gate_name=gate_name, noise_fn=self.default_1q_noise_model
            )
        else:
            raise ValueError(f"No noise model found for gate '{gate_name}'.")

        self._noise_model_instances[gate_name] = noise_model
        return noise_model

    def _get_time_model(self, gate_name: str) -> Optional[TimeModel]:
        """Retrieve the time model for a given gate name."""
        if gate_name in self._time_model_instances:
            return self._time_model_instances[gate_name]

        time_model = None
        if gate_name in self.time_models:
            time_model = TimeModel(
                gate_name=gate_name, time_fn=self.time_models[gate_name]
            )
        elif self.default_1q_time_model and gate_name in _1Q_INSTRUCTIONS:
            time_model = TimeModel(
                gate_name=gate_name, time_fn=self.default_1q_time_model
            )
        else:
            raise ValueError(f"No time model found for gate '{gate_name}'.")

        self._time_model_instances[gate_name] = time_model
        return time_model

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=None
        )
        for inst in _1Q_INSTRUCTIONS:
            yield InstructionProperties(name=inst, params=[], qubits=None)
        yield InstructionProperties(name='cx', params=[], qubits=None)

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        if name not in [*_1Q_INSTRUCTIONS, 'delay', 'cx']:
            raise ValueError(
                f"Instruction '{name}' is not supported by {self.name}."
            )

        time_model = self._get_time_model(name)
        noise_model = self._get_noise_model(name)

        if not time_model or not noise_model:
            raise ValueError(
                f"No time or noise model found for instruction '{name}'."
            )

        duration = time_model.apply(params, self.backend_parameters)
        errors = noise_model.apply(params, self.backend_parameters)

        try:
            quantum_errors = pauli_errors_to_chi(errors)
        except ValueError as e:
            raise ValueError(
                f'The parameters of the processor led to '
                f'inconsistent error probabilities for instruction "{name}"'
            ) from e

        return AppliedInstruction(
            duration=duration,
            quantum_errors=quantum_errors,
            readout_errors=None,
        )
