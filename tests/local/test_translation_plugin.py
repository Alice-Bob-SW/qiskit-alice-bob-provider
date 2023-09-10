from typing import List

import pytest
from qiskit import QuantumCircuit
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.transpiler import PassManager, TranspilerError

from qiskit_alice_bob_provider.local.translation_plugin import (
    BreakDownInitializePass,
    IntToLabelInitializePass,
)


def _check_initialize(circuit: QuantumCircuit, expected: str) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == 1
    assert isinstance(initializes[0].operation, Initialize)
    params = initializes[0].operation.params
    assert params == list(expected)


def test_int_to_label() -> None:
    pm = PassManager([IntToLabelInitializePass()])

    # int(2) should map to '10'
    circ = QuantumCircuit(2)
    circ.initialize(2)
    transpiled = pm.run(circ)
    _check_initialize(transpiled, '10')

    # a string label should pass unchanged
    circ = QuantumCircuit(2)
    circ.initialize('+0')
    transpiled = pm.run(circ)
    _check_initialize(transpiled, '+0')

    # a state vector should fail
    circ = QuantumCircuit(2)
    circ.initialize([1, 0, 0, 0])
    with pytest.raises(TranspilerError):
        pm.run(circ)


def _check_initializes(circuit: QuantumCircuit, expected: List[str]) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == len(expected)
    for initialize in initializes:
        assert len(initialize.qubits) == 1
        qubit = circuit.find_bit(initialize.qubits[0]).index
        assert isinstance(initialize.operation, Initialize)
        assert initialize.operation.params[0] == expected[qubit]


def test_break_down() -> None:
    pm = PassManager([BreakDownInitializePass()])

    # Initialize('10') is broken down into '0' on qubit 0, '1' on qubit 1
    circ = QuantumCircuit(2)
    circ.initialize('10')
    transpiled = pm.run(circ)
    _check_initializes(transpiled, ['0', '1'])

    # Initialize('+') is broken down into '1' on qubit 0
    circ = QuantumCircuit(2)
    circ.initialize('+', 0)
    transpiled = pm.run(circ)
    _check_initializes(transpiled, ['+'])

    # fails on int
    circ = QuantumCircuit(2)
    circ.initialize(2)
    with pytest.raises(TranspilerError):
        pm.run(circ)
