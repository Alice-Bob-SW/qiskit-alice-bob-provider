from typing import Iterator, List, Tuple

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import TranspilerError

from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)

_NOT_SYNTHESIZED = {'initialize', 'measure', 'measure_x', 'delay'}


def _rx_circuit() -> QuantumCircuit:
    circ = QuantumCircuit(1, 1)
    circ.initialize('0', 0)
    circ.rx(0.3, 0)
    circ.measure(0, 0)
    return circ


def _synthesized_gate_count(circ: QuantumCircuit) -> int:
    return sum(
        count
        for name, count in circ.count_ops().items()
        if name not in _NOT_SYNTHESIZED
    )


def test_default_recursion_degree_is_3() -> None:
    backend = AliceBobLocalProvider().get_backend('EMU:40Q:LOGICAL_TARGET')
    default = transpile(_rx_circuit(), backend)
    degree_3 = transpile(
        _rx_circuit(),
        backend,
        unitary_synthesis_plugin_config={'recursion_degree': 3},
    )
    assert default.count_ops() == degree_3.count_ops()
    assert _synthesized_gate_count(default) < 1000


def test_user_recursion_degree_overrides_default() -> None:
    backend = AliceBobLocalProvider().get_backend('EMU:40Q:LOGICAL_TARGET')
    default = transpile(_rx_circuit(), backend)
    degree_2 = transpile(
        _rx_circuit(),
        backend,
        unitary_synthesis_plugin_config={'recursion_degree': 2},
    )
    assert _synthesized_gate_count(degree_2) < _synthesized_gate_count(default)


def test_default_depth_is_5() -> None:
    backend = AliceBobLocalProvider().get_backend('EMU:40Q:LOGICAL_TARGET')
    default = transpile(_rx_circuit(), backend)
    depth_5 = transpile(
        _rx_circuit(), backend, unitary_synthesis_plugin_config={'depth': 5}
    )
    depth_4 = transpile(
        _rx_circuit(), backend, unitary_synthesis_plugin_config={'depth': 4}
    )
    assert default.count_ops() == depth_5.count_ops()
    assert default.count_ops() != depth_4.count_ops()


@pytest.mark.parametrize(
    'name',
    ['EMU:40Q:LOGICAL_TARGET', 'EMU:40Q:LOGICAL_NOISELESS'],
)
def test_unbound_rotation_raises_transpiler_error(name: str) -> None:
    circ = QuantumCircuit(1, 1)
    circ.initialize('0', 0)
    circ.rx(Parameter('theta'), 0)
    circ.measure(0, 0)
    backend = AliceBobLocalProvider().get_backend(name)
    with pytest.raises(
        TranspilerError,
        match=r'rx depends on the unbound parameters theta\. Bind',
    ):
        transpile(circ, backend)


class _DiscreteProcessor(ProcessorDescription):
    def __init__(self, one_qubit_gates: List[str]):
        self.clock_cycle = 1
        self.n_qubits = 2
        self._one_qubit_gates = one_qubit_gates

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for name in [*self._one_qubit_gates, 'cx', 'p0', 'p+', 'mz', 'mx']:
            yield InstructionProperties(name=name, params=[], qubits=None)
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=None
        )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        return AppliedInstruction(
            duration=params[0] if name == 'delay' else 1e3,
            quantum_errors=None,
            readout_errors=None,
        )


@pytest.mark.parametrize('reverse', [False, True])
def test_synthesis_uses_the_basis_of_each_backend(reverse: bool) -> None:
    backends = [
        ProcessorSimulator(
            _DiscreteProcessor(gates), translation_stage_plugin='sk_synthesis'
        )
        for gates in (['h', 't', 'tdg'], ['h', 's', 'sdg', 't', 'x'])
    ]
    if reverse:
        backends.reverse()
    for backend in backends:
        transpiled = transpile(_rx_circuit(), backend)
        assert set(transpiled.count_ops()) <= set(
            backend.target.operation_names
        )
