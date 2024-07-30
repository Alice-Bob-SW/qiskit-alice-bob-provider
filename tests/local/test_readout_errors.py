import pytest
from qiskit import QuantumCircuit

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.local.readout_errors import (
    build_readout_noise_model,
)
from qiskit_alice_bob_provider.processor.description import (
    ProcessorDescription,
)

from .processor_fixture import (
    AllToAllProcessorWithQubitInstruction,
    AllToAllReadoutErrorProcessor,
    ConflictingReadoutErrorsProcessor,
    QubitProcessorWithAllToAllInstruction,
    ReadoutErrorProcessor,
)


def test_one_readout_error() -> None:
    backend = ProcessorSimulator(processor=ReadoutErrorProcessor())

    circ = QuantumCircuit(2, 2)
    circ.initialize('++')
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    counts = backend.run(circ, shots=1).result().get_counts()
    assert counts.keys() == {'01'}

    circ = QuantumCircuit(2, 2)
    circ.initialize('--')
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    counts = backend.run(circ, shots=1).result().get_counts()
    assert counts.keys() == {'01'}


def test_all_to_all_one_readout_error() -> None:
    with pytest.warns(UserWarning):
        backend = ProcessorSimulator(processor=AllToAllReadoutErrorProcessor())

    circ = QuantumCircuit(2, 2)
    circ.initialize('++')
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    counts = backend.run(circ, shots=1).result().get_counts()
    assert counts.keys() == {'11'}

    circ = QuantumCircuit(2, 2)
    circ.initialize('--')
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    counts = backend.run(circ, shots=1).result().get_counts()
    assert counts.keys() == {'11'}


def test_conflicting_readout_errors() -> None:
    proc = ConflictingReadoutErrorsProcessor()
    with pytest.warns(UserWarning):
        backend = ProcessorSimulator(processor=proc)

    circ = QuantumCircuit(1, 1)
    circ.initialize('+')
    circ.measure_x(0, 0)
    counts = backend.run(circ, shots=1).result().get_counts()
    assert counts.keys() == {'1'}

    circ = QuantumCircuit(1, 1)
    circ.initialize('-')
    circ.measure_x(0, 0)
    counts = backend.run(circ, shots=1).result().get_counts()
    assert counts.keys() == {'1'}


@pytest.mark.parametrize(
    'proc',
    [
        AllToAllProcessorWithQubitInstruction(),
        QubitProcessorWithAllToAllInstruction(),
    ],
)
def test_bad_instruction_with_processor_type(
    proc: ProcessorDescription,
) -> None:
    with pytest.raises(ValueError):
        build_readout_noise_model(proc)
