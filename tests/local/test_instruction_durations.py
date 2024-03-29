# pylint: disable=redefined-outer-name

from typing import Iterator, List, Tuple

import pytest
from qiskit.circuit import Barrier, Delay
from qiskit.circuit.library import CCXGate, RXGate, XGate

from qiskit_alice_bob_provider.local.instruction_durations import (
    ProcessorInstructionDurations,
)
from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)


class _TestProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float):
        self.clock_cycle = clock_cycle
        self._delay_qubits: List[Tuple[int, ...]] = [(0,), (1,)]
        self._ccx_qubits: List[Tuple[int, ...]] = [(0, 1, 2), (0, 2, 1)]
        self._rx_qubits: List[Tuple[int, ...]] = [(2,)]

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for tup in self._delay_qubits:
            yield InstructionProperties(
                name='delay', params=['duration'], qubits=tup
            )
        for tup in self._ccx_qubits:
            yield InstructionProperties(name='ccx', params=[], qubits=tup)
        for tup in self._rx_qubits:
            yield InstructionProperties(
                name='rx', params=['angle'], qubits=tup
            )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        if name == 'delay':
            return AppliedInstruction(
                duration=params[0], quantum_errors=None, readout_errors=None
            )
        elif name == 'ccx':
            if qubits == self._ccx_qubits[0]:
                return AppliedInstruction(
                    duration=0.1, quantum_errors=None, readout_errors=None
                )
            else:
                return AppliedInstruction(
                    duration=0.2, quantum_errors=None, readout_errors=None
                )
        elif name == 'rx':
            if qubits == self._rx_qubits[0]:
                return AppliedInstruction(
                    duration=params[0] * 2,
                    quantum_errors=None,
                    readout_errors=None,
                )
        raise NotImplementedError()


class _AllToAllTestProcessor(ProcessorDescription):
    def __init__(self, clock_cycle: float):
        self.clock_cycle = clock_cycle
        self.n_qubits = 3

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=None
        )
        yield InstructionProperties(name='ccx', params=[], qubits=None)
        yield InstructionProperties(name='rx', params=['angle'], qubits=None)

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        if name == 'delay':
            return AppliedInstruction(
                duration=params[0], quantum_errors=None, readout_errors=None
            )
        elif name == 'ccx':
            return AppliedInstruction(
                duration=0.1, quantum_errors=None, readout_errors=None
            )
        elif name == 'rx':
            return AppliedInstruction(
                duration=params[0] * 2,
                quantum_errors=None,
                readout_errors=None,
            )
        raise NotImplementedError()


@pytest.mark.parametrize(
    'proc',
    [
        _TestProcessor(clock_cycle=1e-3),
        _AllToAllTestProcessor(clock_cycle=1e-3),
    ],
)
def test_bad_gate(proc: ProcessorDescription) -> None:
    instr_dur = ProcessorInstructionDurations(proc)
    with pytest.raises(ValueError):  # bad instruction
        instr_dur.get(inst=XGate(), qubits=0)


def test_bad_qubits() -> None:
    proc = _TestProcessor(clock_cycle=1e-3)
    instr_dur = ProcessorInstructionDurations(proc)
    with pytest.raises(ValueError):  # bad qubits
        instr_dur.get(inst=CCXGate(), qubits=(0, 1, 3))


@pytest.mark.parametrize(
    'proc',
    [
        _TestProcessor(clock_cycle=1e-3),
        _AllToAllTestProcessor(clock_cycle=1e-3),
    ],
)
def test_instruction_as_string(proc: ProcessorDescription) -> None:
    instr_dur = ProcessorInstructionDurations(proc)
    assert instr_dur.get(inst='ccx', qubits=(0, 1, 2), unit='s') == 0.1


@pytest.mark.parametrize(
    'proc',
    [
        _TestProcessor(clock_cycle=1e-3),
        _AllToAllTestProcessor(clock_cycle=1e-3),
    ],
)
def test_barrier(proc: ProcessorDescription) -> None:
    instr_dur = ProcessorInstructionDurations(proc)
    assert instr_dur.get(inst='barrier', qubits=(0, 1, 2), unit='s') == 0.0
    assert instr_dur.get(inst=Barrier(3), qubits=(0, 1, 2), unit='s') == 0.0


def test_qubits_forwarded_to_processor() -> None:
    proc = _TestProcessor(clock_cycle=1e-3)
    instr_dur = ProcessorInstructionDurations(proc)
    assert instr_dur.get(inst=CCXGate(), qubits=(0, 1, 2), unit='s') == 0.1
    assert instr_dur.get(inst=CCXGate(), qubits=(0, 2, 1), unit='s') == 0.2


@pytest.mark.parametrize(
    'proc',
    [
        _TestProcessor(clock_cycle=1e-3),
        _AllToAllTestProcessor(clock_cycle=1e-3),
    ],
)
def test_time_unit_conversion(proc: ProcessorDescription) -> None:
    instr_dur = ProcessorInstructionDurations(proc)
    assert instr_dur.get(inst=Delay(2, 's'), qubits=(0,), unit='s') == 2
    assert instr_dur.get(inst=Delay(2, 'dt'), qubits=(0,), unit='s') == 2e-3
    assert instr_dur.get(inst=Delay(2, 'dt'), qubits=(0,), unit='dt') == 2
    with pytest.warns(UserWarning):
        assert instr_dur.get(inst=Delay(2, 'us'), qubits=(0,), unit='dt') == 0
    assert instr_dur.get(inst=Delay(2000, 'us'), qubits=(0,), unit='dt') == 2


@pytest.mark.parametrize(
    'proc',
    [
        _TestProcessor(clock_cycle=1e-3),
        _AllToAllTestProcessor(clock_cycle=1e-3),
    ],
)
def test_instructions_with_params_not_delay(
    proc: ProcessorDescription,
) -> None:
    instr_dur = ProcessorInstructionDurations(proc)
    assert (
        instr_dur.get(inst=RXGate(0.2), qubits=2, parameters=[0.3], unit='s')
        == 0.6
    )
    assert instr_dur.get(inst=RXGate(0.2), qubits=2, unit='s') == 0.4
