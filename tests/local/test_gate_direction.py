from typing import Iterator, List, Optional, Tuple

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ControlFlowOp
from qiskit.transpiler import TranspilerError

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)


class _OneWayCnotProcessor(ProcessorDescription):
    def __init__(self, one_qubit_gates: Tuple[str, ...] = ('h', 'x')):
        self.clock_cycle = 1
        self._one_qubit_gates = one_qubit_gates

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(name='cx', params=[], qubits=(0, 1))
        for qubit in range(2):
            for name in [*self._one_qubit_gates, 'p0', 'mz']:
                yield InstructionProperties(
                    name=name, params=[], qubits=(qubit,)
                )
            yield InstructionProperties(
                name='delay', params=['duration'], qubits=(qubit,)
            )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        return AppliedInstruction(
            duration=params[0] if name == 'delay' else 1e3,
            quantum_errors=None,
            readout_errors=None,
        )


def _cx_pairs(
    circ: QuantumCircuit, indices: Optional[List[int]] = None
) -> List[Tuple[int, ...]]:
    if indices is None:
        indices = [circ.find_bit(q).index for q in circ.qubits]
    pairs = []
    for inst in circ.data:
        qubits = [indices[circ.find_bit(q).index] for q in inst.qubits]
        if isinstance(inst.operation, ControlFlowOp):
            for block in inst.operation.blocks:
                pairs += _cx_pairs(block, qubits)
        elif inst.operation.name == 'cx':
            pairs.append(tuple(qubits))
    return pairs


def _cx_in_if_test() -> QuantumCircuit:
    circ = QuantumCircuit(2, 2)
    circ.measure(0, 0)
    with circ.if_test((circ.clbits[0], 0)):
        circ.cx(1, 0)
    circ.measure([0, 1], [0, 1])
    return circ


@pytest.mark.parametrize('optimization_level', [0, 1, 2, 3])
def test_cnot_is_flipped_to_the_native_direction(
    optimization_level: int,
) -> None:
    circ = QuantumCircuit(2, 2)
    circ.cx(1, 0)
    circ.measure([0, 1], [0, 1])
    transpiled = transpile(
        circ,
        ProcessorSimulator(_OneWayCnotProcessor()),
        optimization_level=optimization_level,
        initial_layout=[0, 1],
    )
    pairs = [
        tuple(transpiled.find_bit(q).index for q in inst.qubits)
        for inst in transpiled.data
        if inst.operation.name == 'cx'
    ]
    assert pairs == [(0, 1)]


def test_cnot_that_the_target_cannot_reverse_raises() -> None:
    circ = QuantumCircuit(2, 2)
    circ.cx(1, 0)
    circ.measure([0, 1], [0, 1])
    with pytest.raises(
        TranspilerError, match=r'cannot reverse the cx on qubits \(1, 0\)'
    ):
        transpile(
            circ,
            ProcessorSimulator(_OneWayCnotProcessor(one_qubit_gates=('x',))),
            optimization_level=0,
            initial_layout=[0, 1],
        )


@pytest.mark.parametrize('optimization_level', [0, 1, 2, 3])
def test_cnot_in_control_flow_block_is_flipped(
    optimization_level: int,
) -> None:
    transpiled = transpile(
        _cx_in_if_test(),
        ProcessorSimulator(_OneWayCnotProcessor()),
        optimization_level=optimization_level,
        initial_layout=[0, 1],
    )
    assert _cx_pairs(transpiled) == [(0, 1)]


def test_cnot_in_control_flow_block_that_the_target_cannot_reverse_raises() -> (
    None
):
    with pytest.raises(
        TranspilerError, match='cannot reverse a cx in a control-flow block'
    ):
        transpile(
            _cx_in_if_test(),
            ProcessorSimulator(_OneWayCnotProcessor(one_qubit_gates=('x',))),
            optimization_level=0,
            initial_layout=[0, 1],
        )


@pytest.mark.parametrize('optimization_level', [0, 1, 2, 3])
def test_cnot_is_flipped_with_the_sk_synthesis_plugin(
    optimization_level: int,
) -> None:
    circ = QuantumCircuit(2, 2)
    circ.cx(1, 0)
    circ.measure([0, 1], [0, 1])
    transpiled = transpile(
        circ,
        ProcessorSimulator(
            _OneWayCnotProcessor(), translation_stage_plugin='sk_synthesis'
        ),
        optimization_level=optimization_level,
        initial_layout=[0, 1],
    )
    assert _cx_pairs(transpiled) == [(0, 1)]
