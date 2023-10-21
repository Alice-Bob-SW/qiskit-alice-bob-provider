import numpy as np
import pytest

from qiskit_alice_bob_provider.processor.utils import (
    compose_1q_errors,
    full_flip_error,
    index_to_pauli_label,
    pauli_errors_to_chi,
    pauli_label_to_index,
    tensor_errors,
)


def test_1_qubit_gate() -> None:
    d = {'X': 0.3, 'Y': 0.2, 'Z': 0.4}
    chi = pauli_errors_to_chi(d)
    diag = np.diag(chi)

    # Check Pauli errors
    assert diag[0] == pytest.approx(0.1)
    assert diag[1] == pytest.approx(0.3)
    assert diag[2] == pytest.approx(0.2)
    assert diag[3] == pytest.approx(0.4)

    # Check off-diagonals are all zeros
    for i in range(chi.shape[0]):
        chi[i, i] = 0
    assert np.all(chi == 0)


def test_2_qubit_gate() -> None:
    d = {
        'XI': 0.3,
        'IY': 0.2,
        'ZY': 0.4,
        'II': 0.2  # II is wrong but this should not matter because it is
        # recomputed anyway by pauli_errors_to_chi
    }
    chi = pauli_errors_to_chi(d)
    diag = np.diag(chi)

    # Check Pauli errors
    assert diag[0] == pytest.approx(0.1)
    assert diag[4] == pytest.approx(0.3)
    assert diag[2] == pytest.approx(0.2)
    assert diag[14] == pytest.approx(0.4)

    # Check off-diagonals are all zeros
    for i in range(chi.shape[0]):
        chi[i, i] = 0
    assert np.all(chi == 0)


def test_bad_probabilities() -> None:
    with pytest.raises(ValueError):
        pauli_errors_to_chi({'XI': 0.5, 'ZY': 0.6})
    with pytest.raises(ValueError):
        pauli_errors_to_chi({'XI': 0.5, 'ZY': -0.1})


def test_bad_label() -> None:
    with pytest.raises(ValueError):
        pauli_errors_to_chi({'A': 0.2})


@pytest.mark.parametrize('label', ['ZI', 'YYZ', 'I', 'XZX'])
def test_label_conversion(label: str) -> None:
    n_qubits = len(label)
    assert label == index_to_pauli_label(n_qubits, pauli_label_to_index(label))


def test_flip_error() -> None:
    # extreme case: one of rx * t, ry * t, rz * t is large,
    # the two others are zero
    for idx in range(3):
        input, output = [0.0] * 3, [0.0] * 3
        input[idx] = 1e10
        output[idx] = 0.5
        assert np.array_equiv(full_flip_error(input), output)

    # extreme case: t is 0
    assert np.array_equiv(full_flip_error([0.0, 0.0, 0.0]), [0, 0, 0])

    # extreme case: all are large
    assert np.array_equiv(
        full_flip_error([1e10, 1e10, 1e10]), [0.25, 0.25, 0.25]
    )

    input_2d = [
        [1e10, 0, 0],
        [0, 1e10, 0],
    ]
    output_2d = [
        [0.5, 0, 0],
        [0, 0.5, 0],
    ]
    assert np.array_equiv(full_flip_error(input_2d), output_2d)

    # extreme case: all are very small, they don't influence each other
    input = [1e-12, 1e-13, 1e-14]
    expected = [1e-12, 1e-13, 1e-14]
    out = full_flip_error(input)
    for i in range(3):
        assert out[i] == pytest.approx(expected[i])


def test_compose_errors() -> None:
    computed = compose_1q_errors({'X': 0.2}, {'Z': 0.6})
    expected = {
        'X': 0.2 * (1.0 - 0.6),
        'Y': 0.2 * 0.6,
        'Z': 0.6 * (1.0 - 0.2),
    }
    assert len(computed) == len(expected)
    for k, v in computed.items():
        assert v == pytest.approx(expected[k])


def test_tensor_errors() -> None:
    computed = tensor_errors({'X': 0.2}, {'YZ': 0.6})
    expected = {
        'IIX': 0.2 * (1.0 - 0.6),
        'YZX': 0.2 * 0.6,
        'YZI': (1.0 - 0.2) * 0.6,
    }
    assert len(computed) == len(expected)
    for k, v in computed.items():
        assert v == pytest.approx(expected[k])
