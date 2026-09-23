from qiskit import QuantumCircuit, transpile

from qiskit_alice_bob_provider.custom_instructions import MeasureX


def test_measure_x_lowers_to_basis_without_measure_x() -> None:
    circ = QuantumCircuit(1, 1)
    circ.append(MeasureX(), [0], [0])
    transpiled = transpile(
        circ,
        basis_gates=['h', 'measure', 'rz', 'sx', 'x', 'cx'],
        optimization_level=0,
    )
    assert [inst.operation.name for inst in transpiled.data] == [
        'h',
        'measure',
        'h',
    ]
