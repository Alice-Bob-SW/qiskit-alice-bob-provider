import numpy as np
import pytest

from qiskit_alice_bob_provider.processor.logical_cat import (
    NOISELESS_CAT_ERROR,
    LogicalCatProcessor,
)


@pytest.mark.parametrize(
    'arguments',
    [
        {'average_nb_photons': -3},
        {'kappa_1': 2},
        {'kappa_1': 100, 'kappa_2': 100_000_000_000},
        {'distance': 0},
        {'distance': 6},
    ],
)
def test_parameter_validation(arguments) -> None:
    with pytest.raises(ValueError):
        LogicalCatProcessor.create_noisy(**arguments)


@pytest.mark.parametrize(
    'arguments',
    [
        {'kappa_1': 10},
        {'kappa_1': 10, 'kappa_2': 10_000},
        {'average_nb_photons': 4},
    ],
)
def test_noisy_parameter_validator_ok(arguments) -> None:
    LogicalCatProcessor.create_noisy(**arguments)


@pytest.mark.parametrize(
    'arguments',
    [
        {'n_qubits': 10},
        {'clock_cycle': 1e-9},
        {},
    ],
)
def test_from_noiseless_parameter_validator_ok(arguments) -> None:
    LogicalCatProcessor.create_noiseless(**arguments)


@pytest.mark.parametrize(
    'arguments',
    [
        {'kappa_1': 10},
        {'kappa_1': 10, 'kappa_2': 10_000},
        {'average_nb_photons': 4},
    ],
)
def test_from_noiseless_parameter_validation_raises(arguments) -> None:
    with pytest.raises(ValueError) as e:
        LogicalCatProcessor.create_noiseless(**arguments)
    assert e.value.args[0] == NOISELESS_CAT_ERROR


@pytest.mark.parametrize(
    'arguments',
    [
        {'kappa_1': 10},
        {'kappa_1': 10, 'kappa_2': 10_000},
        {'average_nb_photons': 4},
    ],
)
def test_noiseless_parameter_validation_raises(arguments) -> None:
    with pytest.raises(ValueError) as e:
        LogicalCatProcessor(**arguments, noiseless=True)
    assert e.value.args[0] == NOISELESS_CAT_ERROR


@pytest.mark.parametrize(
    'get_cat',
    [
        LogicalCatProcessor.create_noisy,
        LogicalCatProcessor.create_noiseless,
    ],
)
def test_all_instructions(get_cat) -> None:
    proc = get_cat()
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
    proc = (
        LogicalCatProcessor.create_noisy()
    )  # Equivalent to LogicalCatProcessor(), better for clarity
    error = proc.apply_instruction('delay', (0,), [1e-8]).quantum_errors
    assert error is not None
    assert error.shape == (4, 4)


def test_delay_instruction_short_noiseless() -> None:
    """The quantum should never be empty, even when the duration is shorter
    than an error correction cycle."""
    proc = LogicalCatProcessor.create_noiseless()
    error = proc.apply_instruction('delay', (0,), [1e-8]).quantum_errors
    assert error is None


def test_1q_instruction() -> None:
    d, nbar, k1, k2 = 5, 16, 100, 10_000_000
    proc = LogicalCatProcessor(
        distance=d, average_nb_photons=nbar, kappa_1=k1, kappa_2=k2
    )
    applied = proc.apply_instruction('x', (0,), [])
    t = 5 * d / k2
    assert applied.duration == pytest.approx(t)
    px = (d - 1) * d * np.exp(-2 * nbar)
    pz = 5.6e-2 * d * (nbar**0.86 * k1 / k2 / 1.3e-2) ** (0.5 * (d + 1))
    assert applied.quantum_errors is not None
    assert applied.readout_errors is None
    diag = np.diag(applied.quantum_errors)
    assert diag[0] == pytest.approx(1.0)  # first order
    assert diag[1] == pytest.approx(px)  # first order
    assert diag[2] == pytest.approx(px * pz)
    assert diag[3] == pytest.approx(pz)  # first order
    assert diag.sum() == pytest.approx(1.0)


def test_1q_instruction_noiseless() -> None:
    # pylint: disable=protected-access
    proc = LogicalCatProcessor.create_noiseless()
    d, _, _, k2 = (
        proc._distance,
        proc._average_nb_photons,
        proc._kappa_1,
        proc._kappa_2,
    )

    applied = proc.apply_instruction('x', (0,), [])
    t = 5 * d / k2
    assert applied.duration == pytest.approx(t)

    # Noiseless Cat: Not quantum or readout errors
    assert applied.quantum_errors is None
    assert applied.readout_errors is None
