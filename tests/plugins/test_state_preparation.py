from typing import List

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, QuantumRegister
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.transpiler import PassManager, PassManagerConfig, TranspilerError

from qiskit_alice_bob_provider.local import AliceBobLocalProvider
from qiskit_alice_bob_provider.plugins.state_preparation import (
    BreakDownInitializePass,
    EnsurePreparationPass,
    IntToLabelInitializePass,
    StatePreparationPlugin,
)


def _assert_mapping_physical_qreg(circuit: QuantumCircuit) -> None:
    """
    Assert if the PassManager correctly mapped the virtual qubits to
    one registry of physical qubits.
    With regards to Qiskit's implementation, a circuit is considered physical
    if it contains one registry named 'q'.
    """
    assert len(circuit.qregs) == 1
    assert circuit.qregs[0].name == 'q'


def _assert_one_initialize(circuit: QuantumCircuit, expected: str) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == 1
    assert isinstance(initializes[0].operation, Initialize)
    params = initializes[0].operation.params
    assert params == list(expected)


def _assert_many_initializes(
    circuit: QuantumCircuit, expected: List[str]
) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == len(expected)
    for initialize in initializes:
        assert len(initialize.qubits) == 1
        qubit = circuit.find_bit(initialize.qubits[0]).index
        assert isinstance(initialize.operation, Initialize)
        assert initialize.operation.params[0] == expected[qubit]


def _assert_gate_in_circuit(
    circuit: QuantumCircuit, expected_gate: str
) -> None:
    gates = circuit.get_instructions(expected_gate)
    assert len(gates) > 0


def test_int_to_label() -> None:
    pm = PassManager([IntToLabelInitializePass()])

    # int(2) should map to '10'
    circ = QuantumCircuit(2)
    circ.initialize(2)
    transpiled = pm.run(circ)
    _assert_one_initialize(transpiled, '10')

    # a string label should pass unchanged
    circ = QuantumCircuit(2)
    circ.initialize('+0')
    transpiled = pm.run(circ)
    _assert_one_initialize(transpiled, '+0')

    # a state vector should fail
    circ = QuantumCircuit(2)
    circ.initialize([1, 0, 0, 0])
    with pytest.raises(TranspilerError):
        pm.run(circ)


def test_break_down() -> None:
    pm = PassManager([BreakDownInitializePass()])

    # Initialize('10') is broken down into '0' on qubit 0, '1' on qubit 1
    circ = QuantumCircuit(2)
    circ.initialize('10')
    transpiled = pm.run(circ)
    _assert_many_initializes(transpiled, ['0', '1'])

    # Initialize('+') is broken down into '1' on qubit 0
    circ = QuantumCircuit(2)
    circ.initialize('+', 0)
    transpiled = pm.run(circ)
    _assert_many_initializes(transpiled, ['+'])

    # fails on int
    circ = QuantumCircuit(2)
    circ.initialize(2)
    with pytest.raises(TranspilerError):
        pm.run(circ)


def test_missing_prep() -> None:
    pm = PassManager([EnsurePreparationPass()])

    c = QuantumCircuit(3, 1)
    c.x(0)
    c.reset(1)
    c.y(1)
    c.initialize('+', 2)
    c.cnot(1, 2)
    c.measure(2, 0)
    new_c = pm.run(c)
    expected = c.count_ops()
    expected['reset'] += 1
    assert dict(new_c.count_ops()) == dict(expected)


def test_enforce_physical_quantum_registry() -> None:
    pm = StatePreparationPlugin().pass_manager(
        PassManagerConfig.from_backend(
            AliceBobLocalProvider().build_logical_backend(n_qubits=4)
        )
    )

    # Circuit with one quantum register of default name 'q'.
    circ = QuantumCircuit(1, 1)
    circ.initialize(1)
    transpiled = pm.run(circ)
    _assert_mapping_physical_qreg(transpiled)

    # Circuit with one quantum register of another name should
    # be renamed to 'q'.
    circ = QuantumCircuit(
        QuantumRegister(size=2, name='foo'),
        ClassicalRegister(size=1, name='bar'),
    )
    circ.initialize(1)
    transpiled = pm.run(circ)
    _assert_mapping_physical_qreg(transpiled)

    # Circuit with more than one quantum register should be merged into
    # a single one named 'q'.
    circ = QuantumCircuit(
        QuantumRegister(size=2, name='foo'),
        QuantumRegister(size=2, name='bar'),
        ClassicalRegister(size=1, name='foobar'),
    )
    circ.initialize(1)
    transpiled = pm.run(circ)
    _assert_mapping_physical_qreg(transpiled)


def test_unroll_custom_definitions() -> None:
    pm = StatePreparationPlugin().pass_manager(
        PassManagerConfig.from_backend(
            AliceBobLocalProvider().build_logical_backend(n_qubits=4)
        )
    )

    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    gate custom q0,q1 { h q0; cx q0,q1; h q1;}
    qreg qf_0[1];
    qreg qf_1[1];
    custom qf_0[0], qf_1[0];
    """
    circ = QuantumCircuit.from_qasm_str(qasm_str)
    assert len(circ) == 1
    _assert_gate_in_circuit(circ, 'custom')

    transpiled = pm.run(circ)
    _assert_gate_in_circuit(transpiled, 'h')
    _assert_gate_in_circuit(transpiled, 'cx')
