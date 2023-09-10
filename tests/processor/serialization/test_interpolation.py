import numpy as np
import pytest

from qiskit_alice_bob_provider.processor.serialization.interpolate import (
    InterpolatedProcessor,
    InterpolationError,
    _build_interpolator,
    _InterpolatedField,
)
from qiskit_alice_bob_provider.processor.serialization.model import (
    SerializedInstruction,
    SerializedProcessor,
)


def test_1d_pauli_interpolation() -> None:
    with open(
        'tests/processor/serialization/data/1d_instruction.json',
        encoding='utf-8',
    ) as f:
        instr = SerializedInstruction.model_validate_json(f.read())
    pauli_interp = _build_interpolator(instr, _InterpolatedField.PAULI)
    assert pauli_interp is not None

    # one of the points in the simulated data
    assert list(pauli_interp(4)) == pytest.approx([0.92, 0.05, 0.0, 0.03])

    # point within the interval
    assert list(pauli_interp(5)) == pytest.approx([0.865, 0.075, 0.01, 0.05])

    # point out of the interval
    with pytest.raises(InterpolationError):
        pauli_interp([7])


def test_1d_duration_interpolation() -> None:
    with open(
        'tests/processor/serialization/data/1d_instruction.json',
        encoding='utf-8',
    ) as f:
        instr = SerializedInstruction.model_validate_json(f.read())
    duration_interp = _build_interpolator(instr, _InterpolatedField.DURATION)
    assert duration_interp is not None

    # one of the points in the simulated data
    assert list(duration_interp(4)) == pytest.approx([1e-4])

    # point within the interval
    assert list(duration_interp(5)) == pytest.approx([5.5e-5])

    # point out of the interval
    with pytest.raises(InterpolationError):
        duration_interp([7])


def test_nd_pauli_interpolation() -> None:
    with open(
        'tests/processor/serialization/data/2d_instruction.json',
        encoding='utf-8',
    ) as f:
        instr = SerializedInstruction.model_validate_json(f.read())
    pauli_interp = _build_interpolator(instr, _InterpolatedField.PAULI)
    assert pauli_interp is not None

    # one of the points in the simulated data
    # the other in the convex hull
    for p in [pauli_interp([6, 1.57]), pauli_interp([5, 1.3])]:
        assert len(p) == 4
        assert sum(p) == pytest.approx(1.0)

    # point out of the convex hull
    with pytest.raises(InterpolationError):
        pauli_interp([42, 42.0])


def test_nd_duration_interpolation() -> None:
    with open(
        'tests/processor/serialization/data/2d_instruction.json',
        encoding='utf-8',
    ) as f:
        instr = SerializedInstruction.model_validate_json(f.read())
    duration_interp = _build_interpolator(instr, _InterpolatedField.DURATION)
    assert duration_interp is not None

    # one of the points in the simulated data
    # the other in the convex hull
    for p in [duration_interp([6, 1.57]), duration_interp([5, 1.3])]:
        assert len(p) == 1

    # point out of the convex hull
    with pytest.raises(InterpolationError):
        duration_interp([42, 42.0])


def test_interpolated_processor_duplicate_instructions() -> None:
    with open(
        'tests/processor/serialization/data/duplicate_instructions.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    with pytest.raises(ValueError):
        InterpolatedProcessor(ser)


def test_interpolated_processor_one_point_no_quantum() -> None:
    with open(
        'tests/processor/serialization/data/one_point_no_quantum.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedProcessor(ser)
    applied = proc.apply_instruction('x', (0,), [])
    assert applied.duration == 1e-4
    assert applied.quantum_errors is None


def test_interpolated_processor_one_point() -> None:
    with open(
        'tests/processor/serialization/data/one_point.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedProcessor(ser)
    applied = proc.apply_instruction('x', (0,), [])
    assert applied.duration == 1e-4
    assert applied.quantum_errors is not None
    assert list(np.diag(applied.quantum_errors)) == pytest.approx(
        [0.92, 0.05, 0.01, 0.02]
    )


def test_interpolated_processor_no_quantum() -> None:
    with open(
        'tests/processor/serialization/data/no_quantum.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedProcessor(ser)
    applied = proc.apply_instruction('x', (0,), [5])
    assert applied.duration == 5.5e-5
    assert applied.quantum_errors is None


def test_interpolated_processor_all_types() -> None:
    with open(
        'tests/processor/serialization/data/all_types.json',
        encoding='utf-8',
    ) as f:
        ser = SerializedProcessor.model_validate_json(f.read())
    proc = InterpolatedProcessor(ser)
    list(proc.all_instructions())
    applied = proc.apply_instruction('delay', (0,), [5, 500])
    assert applied.quantum_errors is not None
    assert applied.readout_errors is None
    applied = proc.apply_instruction('mx', (0,), [5])
    assert applied.quantum_errors is not None
    assert applied.readout_errors is not None
