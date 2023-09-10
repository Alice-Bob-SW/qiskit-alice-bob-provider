import pytest
from qiskit.circuit import Delay, Instruction, Parameter
from qiskit.circuit.library import CXGate, RYGate, XGate
from qiskit.extensions.quantum_initializer import Initialize

from qiskit_alice_bob_provider.custom_instructions import MeasureX
from qiskit_alice_bob_provider.local.proc_to_qiskit import (
    processor_to_qiskit_instruction,
)
from qiskit_alice_bob_provider.processor.description import (
    InstructionProperties,
)


def _compare_instructions(computed: Instruction, expected: Instruction):
    ref_type = type(expected)
    assert isinstance(computed, ref_type)
    assert len(expected.params) == len(computed.params)
    for ref_p, computed_p in zip(expected.params, computed.params):
        assert isinstance(computed_p, type(ref_p))
        if isinstance(ref_p, Parameter):
            assert ref_p.name == computed_p.name
        else:
            assert ref_p == computed_p


def test_compare_instructions() -> None:
    with pytest.raises(AssertionError):
        _compare_instructions(Delay(Parameter('foo')), Delay(Parameter('bar')))
    with pytest.raises(AssertionError):
        _compare_instructions(Delay(23), Delay(Parameter('bar')))
    with pytest.raises(AssertionError):
        _compare_instructions(Delay(Parameter('bar')), Delay(23))
    with pytest.raises(AssertionError):
        _compare_instructions(Delay(24), Delay(23))
    with pytest.raises(AssertionError):
        _compare_instructions(XGate(), Delay(Parameter('bar')))


def test_delay() -> None:
    _compare_instructions(
        computed=processor_to_qiskit_instruction(
            InstructionProperties('delay', (0,), ['foo'])
        ),
        expected=Delay(Parameter('foo')),
    )


def test_initialize() -> None:
    _compare_instructions(
        computed=processor_to_qiskit_instruction(
            InstructionProperties('p+', (0,), [])
        ),
        expected=Initialize('+'),
    )


def test_measure() -> None:
    _compare_instructions(
        computed=processor_to_qiskit_instruction(
            InstructionProperties('mx', (0,), [])
        ),
        expected=MeasureX(),
    )


def test_rotation() -> None:
    _compare_instructions(
        computed=processor_to_qiskit_instruction(
            InstructionProperties('ry', (0,), ['bar'])
        ),
        expected=RYGate(Parameter('bar')),
    )


def test_no_arg_unitaries() -> None:
    _compare_instructions(
        computed=processor_to_qiskit_instruction(
            InstructionProperties('x', (0,), [])
        ),
        expected=XGate(),
    )
    _compare_instructions(
        computed=processor_to_qiskit_instruction(
            InstructionProperties('cx', (0,), [])
        ),
        expected=CXGate(),
    )


def test_unsupported() -> None:
    with pytest.raises(NotImplementedError):
        processor_to_qiskit_instruction(InstructionProperties('u', (0,), []))
