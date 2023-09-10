import numpy as np
import pytest

from qiskit_alice_bob_provider.processor.physical_cat import (
    PhysicalCatProcessor,
)
from qiskit_alice_bob_provider.processor.utils import pauli_label_to_index


def test_bad_coupling_map() -> None:
    with pytest.raises(ValueError):
        PhysicalCatProcessor(coupling_map=[(0, 0)])
    with pytest.raises(ValueError):
        PhysicalCatProcessor(n_qubits=2, coupling_map=[(0, 2)])
    with pytest.raises(ValueError):
        PhysicalCatProcessor(n_qubits=2, coupling_map=[(2, 0)])
    with pytest.raises(ValueError):
        PhysicalCatProcessor(n_qubits=2, coupling_map=[(0, -2)])
    with pytest.raises(ValueError):
        PhysicalCatProcessor(n_qubits=2, coupling_map=[(-2, 0)])


def test_cx_prefactor() -> None:
    proc = PhysicalCatProcessor(
        kappa_1=100, kappa_2=10_000_000, alpha=np.sqrt(19)
    )
    ret = proc.apply_instruction('cx', (0, 1), [])
    terms = ['IX', 'XX', 'XI', 'IY', 'XY', 'XZ']
    s = -np.infty
    assert ret.quantum_errors is not None
    for term in terms:
        idx = pauli_label_to_index(term)
        s = np.logaddexp(s, np.log(ret.quantum_errors[idx, idx]))
    assert np.exp(s) == pytest.approx(0.5 * np.exp(-2 * 19))


def test_idle_prefactor() -> None:
    proc = PhysicalCatProcessor(kappa_1=100, kappa_2=10_000, alpha=np.sqrt(8))
    ret = proc.apply_instruction('delay', (0,), [1e-4])
    terms = ['X', 'Y']
    s = -np.infty
    assert ret.quantum_errors is not None
    for term in terms:
        idx = pauli_label_to_index(term)
        s = np.logaddexp(s, np.log(ret.quantum_errors[idx, idx]))
    assert np.exp(s) == pytest.approx(1e-11)


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        PhysicalCatProcessor(alpha=-3)
    with pytest.raises(ValueError):
        PhysicalCatProcessor(kappa_1=2)
    with pytest.raises(ValueError):
        PhysicalCatProcessor(kappa_1=100, kappa_2=100_000_000_000)
    PhysicalCatProcessor(kappa_1=10)
    PhysicalCatProcessor(kappa_1=10, kappa_2=10_000)
    PhysicalCatProcessor(alpha=2)


def test_unrealistic_probabilities() -> None:
    proc = PhysicalCatProcessor(kappa_1=10_000, alpha=400)
    with pytest.raises(ValueError):
        proc.apply_instruction('cx', (0, 1), [])


def test_all_instructions() -> None:
    proc = PhysicalCatProcessor()
    proc.apply_instruction('mx', (0,), [])
    proc.apply_instruction('mz', (0,), [])
    proc.apply_instruction('delay', (0,), [1e-4])
    proc.apply_instruction('p+', (0,), [])
    proc.apply_instruction('p-', (0,), [])
    proc.apply_instruction('p0', (0,), [])
    proc.apply_instruction('p1', (0,), [])
    proc.apply_instruction('x', (0,), [])
    proc.apply_instruction('rz', (0,), [1.57])
    proc.apply_instruction('z', (0,), [])
    proc.apply_instruction('cx', (0, 1), [])
