import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_alice_bob_provider.local.quantum_errors import (
    build_quantum_error_passes,
)
from qiskit_alice_bob_provider.processor.description import (
    ProcessorDescription,
)

from .processor_fixture import (
    AllToAllProcessorWithQubitInstruction,
    QubitProcessorWithAllToAllInstruction,
    SimpleAllToAllProcessor,
    SimpleProcessor,
)


def test_noise_not_on_all_qubits() -> None:
    circ = QuantumCircuit(3)
    for i in range(3):
        circ.y(i)
    pm = PassManager(build_quantum_error_passes(SimpleProcessor()))
    transpiled = pm.run(circ)

    # Y noise was only activated on qubits 0 and 1
    y_errors = transpiled.get_instructions('y_error')
    assert len(y_errors) == 2
    qubits = {circ.find_bit(e.qubits[0])[0] for e in y_errors}
    assert qubits == {0, 1}


@pytest.mark.parametrize(
    'proc', [SimpleProcessor(), SimpleAllToAllProcessor()]
)
def test_noise_on_initialize(proc: ProcessorDescription) -> None:
    circ = QuantumCircuit(3)
    circ.initialize('+', 0)
    circ.initialize('-', 1)
    circ.initialize('0', 2)
    pm = PassManager(build_quantum_error_passes(proc))
    transpiled = pm.run(circ)

    for label, n_qubits, index in [
        ('p+_error', 1, 0),
        ('p-_error', 0, 0),
        ('p0_error', 1, 2),
        ('p1_error', 0, 0),
    ]:
        errors = transpiled.get_instructions(label)
        assert len(errors) == n_qubits
        if n_qubits == 1:
            qubit = circ.find_bit(errors[0].qubits[0]).index
            assert qubit == index


@pytest.mark.parametrize(
    'proc', [SimpleProcessor(), SimpleAllToAllProcessor()]
)
def test_noise_on_mx(proc: ProcessorDescription) -> None:
    circ = QuantumCircuit(3, 3)
    circ.measure_x(0, 0)
    circ.measure_x(1, 1)
    circ.measure(2, 2)
    pm = PassManager(build_quantum_error_passes(proc))
    transpiled = pm.run(circ)

    # Only qubits 0 and 1 had a measure in the x basis
    mx_errors = transpiled.get_instructions('mx_error')
    assert len(mx_errors) == 2
    qubits = {circ.find_bit(e.qubits[0])[0] for e in mx_errors}
    assert qubits == {0, 1}


@pytest.mark.parametrize(
    'proc', [SimpleProcessor(), SimpleAllToAllProcessor()]
)
def test_pass_with_none_chi_matrix(proc: ProcessorDescription) -> None:
    circ = QuantumCircuit(10)
    circ.x(0)
    pm = PassManager(build_quantum_error_passes(proc))
    transpiled = pm.run(circ)
    assert len(transpiled.get_instructions('x_error')) == 0


@pytest.mark.parametrize(
    'proc', [SimpleProcessor(), SimpleAllToAllProcessor()]
)
def test_chi_vs_pauli_errors(proc: ProcessorDescription) -> None:
    pm = PassManager(build_quantum_error_passes(proc))

    # Circuit with diagonal chi matrix -> quantum channel of Pauli errors
    circ = QuantumCircuit(10)
    circ.y(0)
    transpiled = pm.run(circ)
    assert len(transpiled.decompose().get_instructions('quantum_channel')) == 1

    # Circuit with none diagonal chi matrix -> Kraus map
    circ = QuantumCircuit(10)
    circ.h(0)
    transpiled = pm.run(circ)
    print(transpiled.decompose())
    assert len(transpiled.decompose().get_instructions('kraus')) == 1


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
        PassManager(build_quantum_error_passes(proc))
