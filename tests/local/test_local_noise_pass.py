from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from qiskit_alice_bob_provider.local.patch.local_noise_pass import (
    LocalNoisePass,
)


def test_noise_function_gets_outer_qubits_inside_control_flow() -> None:
    seen = []

    def record(inst, qubits):
        seen.append((inst.name, list(qubits)))

    circ = QuantumCircuit(3, 1)
    circ.measure(0, 0)
    with circ.if_test((circ.clbits[0], 1)):
        circ.cx(2, 1)

    LocalNoisePass(record).run(circuit_to_dag(circ))

    assert ('cx', [2, 1]) in seen
