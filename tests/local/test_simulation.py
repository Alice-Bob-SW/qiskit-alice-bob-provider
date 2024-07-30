from typing import Iterator, List, Set, Tuple

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer.backends import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.local.job import ProcessorSimulationJob
from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)
from qiskit_alice_bob_provider.processor.interpolated_cat import (
    InterpolatedCatProcessor,
)
from qiskit_alice_bob_provider.processor.serialization.model import (
    SerializedProcessor,
)
from qiskit_alice_bob_provider.processor.utils import pauli_errors_to_chi

from .processor_fixture import (
    LargeSimpleProcessor,
    OneQubitProcessor,
    SimpleAllToAllProcessor,
    SimpleProcessor,
)


def gen_circuits() -> Iterator[Tuple[str, QuantumCircuit, Set[str]]]:
    circ = QuantumCircuit(2, 2)
    circ.initialize('0', 0)
    circ.initialize('+', 1)
    circ.x(0)
    circ.cx(0, 1)
    circ.measure(0, 0)
    circ.measure_x(1, 1)
    yield ('everything_circuit', circ, {'11'})


@pytest.mark.parametrize(
    ['tup', 'backend'],
    (
        (tup, backend)
        for tup in gen_circuits()
        for backend in [
            ProcessorSimulator(SimpleProcessor()),
            ProcessorSimulator(
                SimpleAllToAllProcessor(),
                translation_stage_plugin='sk_synthesis',
            ),
        ]
    ),
)
def test_circuit(
    tup: Tuple[str, QuantumCircuit, Set[str]], backend: ProcessorSimulator
) -> None:
    _, circ, expected_keys = tup
    job: ProcessorSimulationJob = backend.run(transpile(circ, backend))
    result = job.result()
    try:
        assert result.get_counts().keys() == expected_keys
    except AssertionError:
        print('==== Original circuit ====')
        print(circ)
        print('==== Scheduled circuit ====')
        print(job.circuits()[0])
        print('==== Noisy circuit ====')
        print(job.noisy_circuits()[0])
        raise


def test_multiple_experiments() -> None:
    circ1 = QuantumCircuit(1, 1)
    circ1.x(0)
    circ1.measure(0, 0)
    circ2 = QuantumCircuit(1, 1)
    circ2.initialize('+')
    circ2.measure_x(0, 0)
    backend = ProcessorSimulator(SimpleProcessor())
    job: ProcessorSimulationJob = backend.run(
        transpile([circ1, circ2], backend)
    )
    result = job.result()
    assert len(result.get_counts()) == 2


def test_non_default_shots() -> None:
    circ = QuantumCircuit(1, 1)
    circ.x(0)
    circ.measure(0, 0)
    shots = 5
    backend = ProcessorSimulator(SimpleProcessor())
    job: ProcessorSimulationJob = backend.run(
        transpile(circ, backend), shots=shots
    )
    result = job.result()
    assert sum(result.get_counts().values()) == shots


class _CXProcessor(ProcessorDescription):
    clock_cycle = 1

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for i in range(2):
            yield InstructionProperties(name='p0', params=[], qubits=(i,))
            yield InstructionProperties(name='mz', params=[], qubits=(i,))
        yield InstructionProperties(name='cx', params=[], qubits=(0, 1))
        yield InstructionProperties(name='cx', params=[], qubits=(1, 0))

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        if name == 'cx':
            return AppliedInstruction(
                duration=self.clock_cycle,
                quantum_errors=pauli_errors_to_chi({'IX': 1.0}),
                readout_errors=None,
            )
        return AppliedInstruction(
            duration=self.clock_cycle,
            quantum_errors=None,
            readout_errors=None,
        )


def gen_circuits_ordering() -> Iterator[Tuple[str, QuantumCircuit]]:
    circ = QuantumCircuit(2, 2)
    circ.cx(0, 1)
    circ.measure(0, 0)
    circ.measure(1, 1)
    yield ('01_circuit', circ)

    circ = QuantumCircuit(2, 2)
    circ.cx(1, 0)
    circ.measure(0, 0)
    circ.measure(1, 1)
    yield ('10_circuit', circ)


@pytest.mark.parametrize('tup', gen_circuits_ordering())
def test_qubit_ordering(tup: Tuple[str, QuantumCircuit]) -> None:
    _, circ = tup
    aer = AerSimulator()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(pauli_error([('IX', 1.0)]), 'cx')
    aer_counts = (
        aer.run(transpile(circ, aer), noise_model=nm).result().get_counts()
    )
    backend = ProcessorSimulator(_CXProcessor())
    proc_counts = backend.run(transpile(circ, backend)).result().get_counts()
    try:
        assert aer_counts == proc_counts
    except AssertionError:
        print(circ)


def test_interpolated_cat() -> None:
    with open(
        'tests/processor/serialization/data/all_types.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedCatProcessor(ser, alpha=np.sqrt(5))
    backend = ProcessorSimulator(proc)

    circ = QuantumCircuit(2, 2)
    circ.initialize('+', 0)
    circ.initialize('0', 1)
    circ.x(0)
    circ.delay(800, 0, unit='s')
    circ.y(0)
    circ.rz(1.57, 0)
    circ.cx(0, 1)
    circ.measure_x(0, 0)
    circ.measure(0, 1)

    job = backend.run(transpile(circ, backend))
    assert isinstance(job, ProcessorSimulationJob)
    noisy_circ = job.noisy_circuits()[0]

    assert len(noisy_circ.get_instructions('p0_error')) == 1
    assert len(noisy_circ.get_instructions('p+_error')) == 1
    assert len(noisy_circ.get_instructions('mx_error')) == 1
    assert len(noisy_circ.get_instructions('x_error')) == 1
    assert len(noisy_circ.get_instructions('y_error')) == 0
    assert len(noisy_circ.get_instructions('rz_error')) == 1
    assert len(noisy_circ.get_instructions('cx_error')) == 1
    assert len(noisy_circ.get_instructions('mx_error')) == 1
    assert len(noisy_circ.get_instructions('mz_error')) == 1
    assert len(noisy_circ.get_instructions('delay')) == 1

    # The next lines test that we can simulate without errors
    job.result().get_counts()


def test_no_delay() -> None:
    """This test makes sure that quantum errors get inserted even when there
    are no delay in the circuit after scheduling."""
    backend = ProcessorSimulator(OneQubitProcessor())
    circ = QuantumCircuit(1, 1)
    circ.initialize('+')
    circ.measure_x(0, 0)
    assert backend.run(
        transpile(circ, backend), shots=1
    ).result().get_counts() == {'1': 1}


class _ConditioningProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float = 1):
        self.clock_cycle = clock_cycle
        self.n_qubits = 2

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(name='p+', params=[], qubits=None)
        yield InstructionProperties(name='p-', params=[], qubits=None)
        yield InstructionProperties(name='mx', params=[], qubits=None)
        yield InstructionProperties(name='x', params=[], qubits=None)

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
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
        elif name == 'x':
            return AppliedInstruction(
                duration=1e3,
                quantum_errors=pauli_errors_to_chi({'Z': 1.0}),
                readout_errors=None,
            )
        raise NotImplementedError()


def test_conditional_instruction() -> None:
    # Noiseless backend, except the X gate that has a 100% Z-flip error
    backend = ProcessorSimulator(_ConditioningProcessor())

    circ = QuantumCircuit(2, 2)
    circ.initialize('++')
    circ.measure_x(0, 0)
    circ.x(1).c_if(0, 0)
    circ.measure_x(1, 1)
    assert backend.run(
        transpile(circ, backend), shots=1
    ).result().get_counts() == {'10': 1}

    circ = QuantumCircuit(2, 2)
    circ.initialize('+-')
    circ.measure_x(0, 0)
    circ.x(1).c_if(0, 0)
    circ.measure_x(1, 1)
    job = backend.run(transpile(circ, backend), shots=1)
    assert job.result().get_counts() == {'01': 1}


def test_large_processor() -> None:
    """A processor with more qubits (40 here) than accepted by AerSimulator
    (29 on my machine, this is based on system memory) would fail. A fix was
    implemented in local/backend.py"""
    backend = ProcessorSimulator(LargeSimpleProcessor())
    circ = QuantumCircuit(1, 1)
    circ.initialize('0')
    circ.delay(1, 0, unit='s')
    circ.measure(0, 0)
    backend.run(transpile(circ, backend))
