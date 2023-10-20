from typing import Iterator, List, Tuple

from qiskit_alice_bob_provider.local.instruction_durations import (
    ProcessorInstructionDurations,
)
from qiskit_alice_bob_provider.local.target import processor_to_target
from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)


class _TestProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float):
        self.clock_cycle = clock_cycle

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=(0,)
        )
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=(1,)
        )
        yield InstructionProperties(name='ccx', params=[], qubits=(0, 1, 2))
        yield InstructionProperties(name='ccx', params=[], qubits=(0, 2, 1))

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        raise NotImplementedError()


class _AllToAllTestProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float):
        self.clock_cycle = clock_cycle
        self.n_qubits = 3

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=None
        )
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=None
        )
        yield InstructionProperties(name='ccx', params=[], qubits=None)
        yield InstructionProperties(name='ccx', params=[], qubits=None)

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        raise NotImplementedError()


def test_target_from_processor() -> None:
    clock_cycle = 3
    target = processor_to_target(_TestProcessor(clock_cycle))
    assert target.dt == clock_cycle
    assert target.instruction_supported('delay', qargs=(0,))
    assert target.instruction_supported('delay', qargs=(1,))
    assert not target.instruction_supported('delay', qargs=(2,))
    assert target.instruction_supported('ccx', qargs=(0, 1, 2))
    assert target.instruction_supported('ccx', qargs=(0, 2, 1))
    assert not target.instruction_supported('ccx', qargs=(3, 4, 5))
    assert isinstance(target.durations(), ProcessorInstructionDurations)


def test_target_from_all_to_all_processor() -> None:
    clock_cycle = 3
    target = processor_to_target(_AllToAllTestProcessor(clock_cycle))
    assert target.dt == clock_cycle
    for i in range(2):
        assert target.instruction_supported('delay', qargs=(i,))
    for i, j, k in zip(range(2), range(2), range(2)):
        if i == j or j == k or i == k:
            continue
        assert target.instruction_supported('ccx', qargs=(i, j, k))
    assert isinstance(target.durations(), ProcessorInstructionDurations)
