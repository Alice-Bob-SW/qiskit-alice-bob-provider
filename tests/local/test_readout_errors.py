import pytest
from qiskit import QuantumCircuit, execute

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator

from .processor_fixture import (
    ConflictingReadoutErrorsProcessor,
    ReadoutErrorProcessor,
)


def test_one_readout_error() -> None:
    proc = ReadoutErrorProcessor()
    backend = ProcessorSimulator(processor=proc)

    circ = QuantumCircuit(2, 2)
    circ.initialize('++')
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    counts = execute(circ, backend, shots=1).result().get_counts()
    assert counts.keys() == {'01'}

    circ = QuantumCircuit(2, 2)
    circ.initialize('--')
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    counts = execute(circ, backend, shots=1).result().get_counts()
    assert counts.keys() == {'01'}


def test_conflicting_readout_errors() -> None:
    proc = ConflictingReadoutErrorsProcessor()
    with pytest.warns(UserWarning):
        backend = ProcessorSimulator(processor=proc)

    circ = QuantumCircuit(1, 1)
    circ.initialize('+')
    circ.measure_x(0, 0)
    counts = execute(circ, backend, shots=1).result().get_counts()
    assert counts.keys() == {'1'}

    circ = QuantumCircuit(1, 1)
    circ.initialize('-')
    circ.measure_x(0, 0)
    counts = execute(circ, backend, shots=1).result().get_counts()
    assert counts.keys() == {'1'}
