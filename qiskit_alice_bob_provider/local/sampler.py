from qiskit.primitives import BackendSamplerV2
from qiskit import QuantumCircuit
from typing import Optional, Sequence, Union


class PreprocessingSampler(BackendSamplerV2):
    """
    A wrapper around BackendSamplerV2 that applies the ProcessorSimulator's
    noise model and gate decomposition before submitting to run().
    """

    def run(
        self,
        circuits: Union[QuantumCircuit, Sequence[QuantumCircuit]],
        parameter_values: Optional[Sequence[Sequence[float]]] = None,
        **run_options,
    ):
        # Ensure we always work with a list of circuits
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]

        # Apply noise and gate decomposition
        preprocessed = [
            self._backend._noise_pass_manager.run([circ])[0]
            .decompose()
            .decompose(gates_to_decompose=['state_preparation'])
            for circ in circuits
        ]

        # Optional: safety check
        for circ in preprocessed:
            for inst, *_ in circ.data:
                if inst.name == 'Q':
                    raise RuntimeError(
                        "Instruction 'Q' was not removed during preprocessing"
                    )

        return super().run(preprocessed, parameter_values, **run_options)
