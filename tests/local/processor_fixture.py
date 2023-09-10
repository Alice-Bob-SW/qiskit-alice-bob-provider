from itertools import product
from typing import Iterator, Tuple

from qiskit.quantum_info.operators import Chi
from qiskit_aer.noise import amplitude_damping_error

from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)
from qiskit_alice_bob_provider.processor.utils import pauli_errors_to_chi


class SimpleProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float = 1):
        self.clock_cycle = clock_cycle

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for i, j in product(range(3), range(3)):
            if i != j:
                yield InstructionProperties(
                    name='cx', params=[], qubits=(i, j)
                )
        for i in range(3):
            yield InstructionProperties(name='h', params=[], qubits=(i,))
            yield InstructionProperties(
                name='rz', params=['angle'], qubits=(i,)
            )
            yield InstructionProperties(name='x', params=[], qubits=(i,))
            if i != 2:
                yield InstructionProperties(name='y', params=[], qubits=(i,))
            yield InstructionProperties(name='p+', params=[], qubits=(i,))
            yield InstructionProperties(name='p0', params=[], qubits=(i,))
            yield InstructionProperties(name='p1', params=[], qubits=(i,))
            yield InstructionProperties(name='mx', params=[], qubits=(i,))
            yield InstructionProperties(name='mz', params=[], qubits=(i,))
            yield InstructionProperties(
                name='delay', params=['duration'], qubits=(i,)
            )

    def apply_instruction(  # pylint: disable=too-many-return-statements
        self, name: str, qubits: Tuple[int, ...], params: list[float]
    ) -> AppliedInstruction:
        if name == 'delay':
            return AppliedInstruction(
                duration=params[0],
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        elif name == 'x':
            return AppliedInstruction(
                duration=1e7,
                quantum_errors=None,
                readout_errors=None,
            )
        elif name == 'y':
            return AppliedInstruction(
                duration=1e7,
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        elif name == 'rz':
            return AppliedInstruction(
                duration=params[0] * 1e6,
                quantum_errors=pauli_errors_to_chi({'X': 1.0}),
                readout_errors=None,
            )
        elif name == 'mz':
            return AppliedInstruction(
                duration=1e5,
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        elif name == 'mx':
            return AppliedInstruction(
                duration=1e4,
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        elif name == 'p0':
            return AppliedInstruction(
                duration=1e3,
                quantum_errors=pauli_errors_to_chi({'X': 1.0}),
                readout_errors=None,
            )
        elif name == 'p1':
            return AppliedInstruction(
                duration=1e3,
                quantum_errors=pauli_errors_to_chi({'X': 1.0}),
                readout_errors=None,
            )
        elif name == 'p+':
            return AppliedInstruction(
                duration=1e2,
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        elif name == 'cx':
            return AppliedInstruction(
                duration=1e1,
                quantum_errors=pauli_errors_to_chi({'IX': 1.0}),
                readout_errors=None,
            )
        elif name == 'h':
            m = Chi(amplitude_damping_error(0.1)).data * 0.5
            return AppliedInstruction(
                duration=1e0,
                quantum_errors=m,
                readout_errors=None,
            )
        raise NotImplementedError()


class ReadoutErrorProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float = 1):
        self.clock_cycle = clock_cycle

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for i in range(2):
            yield InstructionProperties(name='p+', params=[], qubits=(i,))
            yield InstructionProperties(name='p-', params=[], qubits=(i,))
        yield InstructionProperties(
            name='mx', params=[], qubits=(0,), readout_errors=[1, 0]
        )
        yield InstructionProperties(
            name='mx', params=[], qubits=(1,), readout_errors=[0, 1]
        )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: list[float]
    ) -> AppliedInstruction:
        if name == 'mx':
            return AppliedInstruction(
                duration=1e4,
                quantum_errors=None,
                readout_errors=None,
            )
        elif name in {'p+', 'p-'}:
            return AppliedInstruction(
                duration=1e2,
                quantum_errors=None,
                readout_errors=None,
            )
        raise NotImplementedError()


class ConflictingReadoutErrorsProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float = 1):
        self.clock_cycle = clock_cycle

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(name='p+', params=[], qubits=(0,))
        yield InstructionProperties(name='p-', params=[], qubits=(0,))
        yield InstructionProperties(
            name='mx', params=[], qubits=(0,), readout_errors=[1, 0]
        )
        yield InstructionProperties(
            name='mz', params=[], qubits=(0,), readout_errors=[0, 1]
        )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: list[float]
    ) -> AppliedInstruction:
        if name == 'mx':
            return AppliedInstruction(
                duration=1e4,
                quantum_errors=None,
                readout_errors=None,
            )
        elif name in {'p+', 'p-'}:
            return AppliedInstruction(
                duration=1e2,
                quantum_errors=None,
                readout_errors=None,
            )
        raise NotImplementedError()


class OneQubitProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float = 1):
        self.clock_cycle = clock_cycle

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(name='p+', params=[], qubits=(0,))
        yield InstructionProperties(name='mx', params=[], qubits=(0,))

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: list[float]
    ) -> AppliedInstruction:
        if name == 'mx':
            return AppliedInstruction(
                duration=1e4,
                quantum_errors=None,
                readout_errors=None,
            )
        elif name == 'p+':
            return AppliedInstruction(
                duration=1e2,
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        raise NotImplementedError()
