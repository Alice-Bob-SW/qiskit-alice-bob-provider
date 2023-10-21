import numpy as np
import pytest

from qiskit_alice_bob_provider.processor.logical_cat import LogicalCatProcessor


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        LogicalCatProcessor(alpha=-3)
    with pytest.raises(ValueError):
        LogicalCatProcessor(kappa_1=2)
    with pytest.raises(ValueError):
        LogicalCatProcessor(kappa_1=100, kappa_2=100_000_000_000)
    with pytest.raises(ValueError):
        LogicalCatProcessor(distance=0)
    with pytest.raises(ValueError):
        LogicalCatProcessor(distance=6)
    LogicalCatProcessor(kappa_1=10)
    LogicalCatProcessor(kappa_1=10, kappa_2=10_000)
    LogicalCatProcessor(alpha=2)


def test_all_instructions() -> None:
    proc = LogicalCatProcessor()
    proc.apply_instruction('mx', (0,), [])
    proc.apply_instruction('mz', (0,), [])
    proc.apply_instruction('delay', (0,), [1e-4])
    proc.apply_instruction('p+', (0,), [])
    proc.apply_instruction('p-', (0,), [])
    proc.apply_instruction('p0', (0,), [])
    proc.apply_instruction('p1', (0,), [])
    proc.apply_instruction('x', (0,), [])
    proc.apply_instruction('z', (0,), [])
    proc.apply_instruction('h', (0,), [])
    proc.apply_instruction('t', (0,), [])
    proc.apply_instruction('cx', (0, 1), [])
    proc.apply_instruction('ccx', (0, 1, 2), [])


def test_delay_instruction_short() -> None:
    """The quantum should never be empty, even when the duration is shorter
    than an error correction cycle."""
    proc = LogicalCatProcessor()
    error = proc.apply_instruction('delay', (0,), [1e-8]).quantum_errors
    assert error is not None
    assert error.shape == (4, 4)


def test_1q_instruction() -> None:
    d, alpha, k1, k2 = 5, 4, 100, 10_000_000
    proc = LogicalCatProcessor(distance=d, alpha=alpha, kappa_1=k1, kappa_2=k2)
    applied = proc.apply_instruction('x', (0,), [])
    t = 5 * d / k2
    assert applied.duration == pytest.approx(t)
    px = (d - 1) * d * np.exp(-2 * alpha**2)
    pz = (
        5.6e-2
        * d
        * (np.abs(alpha) ** (2 * 0.86) * k1 / k2 / 1.3e-2) ** (0.5 * (d + 1))
    )
    assert applied.quantum_errors is not None
    assert applied.readout_errors is None
    diag = np.diag(applied.quantum_errors)
    assert diag[0] == pytest.approx(1.0)  # first order
    assert diag[1] == pytest.approx(px)  # first order
    assert diag[2] == pytest.approx(px * pz)
    assert diag[3] == pytest.approx(pz)  # first order
    assert diag.sum() == pytest.approx(1.0)
