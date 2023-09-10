from typing import Iterator, Tuple

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.extensions.quantum_initializer import Initialize

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator

from .processor_fixture import SimpleProcessor


def gen_circuits() -> Iterator[Tuple[str, QuantumCircuit, int]]:
    circ = QuantumCircuit(2, 2)
    circ.initialize('0', 0)
    circ.initialize('+', 1)
    circ.x(0)
    circ.rz(11, 1)
    circ.cx(0, 1)
    circ.measure(0, 0)
    circ.measure_x(1, 1)
    yield ('qubits_competing_for_duration', circ, 11100110)

    sub_sub_circ = QuantumCircuit(2, 0)
    sub_sub_circ.x(0)
    sub_sub_circ.rz(11, 1)
    sub_circ = QuantumCircuit(2, 0)
    sub_circ.append(sub_sub_circ.to_instruction(), [1, 0])
    circ = QuantumCircuit(2, 2)
    circ.initialize('0', 0)
    circ.initialize('+', 1)
    circ.append(sub_circ.to_instruction(), [1, 0])
    circ.cx(0, 1)
    circ.measure(0, 0)
    circ.measure_x(1, 1)
    yield ('sub_circuits', circ, 11100110)


@pytest.mark.parametrize('tup', gen_circuits())
def test_circuit(tup: Tuple[str, QuantumCircuit, int]) -> None:
    _, circ, expected_duration = tup
    backend = ProcessorSimulator(SimpleProcessor(1))
    transpiled = transpile(circ, backend)
    try:
        assert transpiled.duration == expected_duration
    except AssertionError:
        print('==== Original circuit ====')
        print(circ)
        print('==== Transpiled circuit ====')
        print(transpiled)
        raise


def test_reset_to_initialize() -> None:
    circ = QuantumCircuit(1, 0)
    circ.reset(0)
    backend = ProcessorSimulator(SimpleProcessor(1))
    transpiled = transpile(circ, backend)
    initializes = transpiled.get_instructions('initialize')
    print(circ)
    print(transpiled)
    assert len(initializes) == 1
    assert isinstance(initializes[0].operation, Initialize)
    assert initializes[0].operation.params[0] == '0'
    assert len(transpiled.get_instructions('reset')) == 0
