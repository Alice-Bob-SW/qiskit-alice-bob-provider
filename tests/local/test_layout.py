import pytest
from qiskit import QuantumCircuit, transpile

from qiskit_alice_bob_provider import AliceBobLocalProvider


def _narrow_circuit_needing_routing() -> QuantumCircuit:
    circ = QuantumCircuit(3, 3)
    circ.initialize('000', range(3))
    circ.cx(0, 2)
    circ.measure(range(3), range(3))
    return circ


@pytest.mark.parametrize('optimization_level', [0, 1, 2, 3])
def test_routed_circuit_respects_coupling_map(optimization_level: int) -> None:
    backend = AliceBobLocalProvider().get_backend('EMU:6Q:PHYSICAL_CATS')
    transpiled = transpile(
        _narrow_circuit_needing_routing(),
        backend,
        optimization_level=optimization_level,
    )
    edges = set(backend.coupling_map.get_edges())
    for inst in transpiled.data:
        if inst.operation.name == 'cx':
            pair = tuple(transpiled.find_bit(q).index for q in inst.qubits)
            assert pair in edges


@pytest.mark.parametrize('optimization_level', [0, 1, 2, 3])
def test_layout_matches_measured_qubits(optimization_level: int) -> None:
    backend = AliceBobLocalProvider().get_backend('EMU:6Q:PHYSICAL_CATS')
    transpiled = transpile(
        _narrow_circuit_needing_routing(),
        backend,
        optimization_level=optimization_level,
    )
    final_layout = transpiled.layout.final_index_layout()
    for inst in transpiled.data:
        if inst.operation.name == 'measure':
            physical = transpiled.find_bit(inst.qubits[0]).index
            virtual = transpiled.find_bit(inst.clbits[0]).index
            assert physical == final_layout[virtual]
