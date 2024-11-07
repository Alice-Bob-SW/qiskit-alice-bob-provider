##############################################################################
# Copyright 2023 Alice & Bob
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from inspect import signature
from itertools import product
from typing import Dict, List, Union

import numpy as np
from typing_extensions import Type


def pauli_errors_to_chi(pauli_errors: Dict[str, float]) -> np.ndarray:
    """Convert a collection of Pauli errors into a quantum process tomography
    Chi matrix.

    https://qiskit.org/documentation/stubs/qiskit.quantum_info.Chi.html

    Args:
        pauli_errors (Dict[str, float]): a dict of Pauli errors where the key
        is the type of Pauli error as a string (e.g., 'I', 'X', 'XI', 'XZX')
        and the value is the probability of this Pauli error.
        The "no error" case (e.g., 'I', 'II', 'III', etc) should not be
        provided as it will be computed automatically by difference.


    Returns:
        np.ndarray: a quantum process tomography Chi matrix. The Qiskit link
        above is a good description of a Chi matrix.
    """
    return np.diag(pauli_errors_to_chi_diag(pauli_errors))


def pauli_errors_to_chi_diag(pauli_errors: Dict[str, float]) -> np.ndarray:
    """Convert a collection of Pauli errors into a quantum process tomography
    Chi matrix.

    https://qiskit.org/documentation/stubs/qiskit.quantum_info.Chi.html

    Args:
        pauli_errors (Dict[str, float]): a dict of Pauli errors where the key
        is the type of Pauli error as a string (e.g., 'I', 'X', 'XI', 'XZX')
        and the value is the probability of this Pauli error.
        The "no error" case (e.g., 'I', 'II', 'III', etc) should not be
        provided as it will be computed automatically by difference.


    Returns:
        np.ndarray: the diagonal of a quantum process tomography Chi matrix.
        The Qiskit link above is a good description of a Chi matrix.
    """
    if len(pauli_errors) == 0:
        raise ValueError(
            'An empty dict of Pauli errors cannot be converted to chi matrix '
            '(need at least one error, even with probability 0, to determine '
            'the number of qubits of the instruction)'
        )
    some_pauli_label = next(iter(pauli_errors))
    n_qubits = len(some_pauli_label)
    diag = np.zeros(shape=(4**n_qubits,), dtype=float)
    for pauli_label, prob in pauli_errors.items():
        diag[pauli_label_to_index(pauli_label)] = prob

    # Computation of the probability of the "no error" case by difference
    diag[0] = 1.0 - np.sum(diag[1:])

    # Sanity check that all probabilities are between 0 and 1
    try:
        assert np.all(diag >= 0)
    except AssertionError as e:
        raise ValueError(
            'Pauli error probabilities are not in [0, 1] or sum up to more'
            f' than 1. Probabilities: {pauli_errors}'
        ) from e

    return diag


def is_diagonal(m: np.ndarray) -> bool:
    """Test if a matrix is diagonal"""
    return np.count_nonzero(m - np.diag(np.diag(m))) == 0


def chi_to_pauli_errors(chi: np.ndarray) -> Dict[str, float]:
    """Convert a quantum process tomography Chi matrix into a collection of
    Pauli errors.

    https://qiskit.org/documentation/stubs/qiskit.quantum_info.Chi.html

    Args:
        chi (np.ndarray): a quantum process tomography Chi matrix. The Qiskit
        link above is a good description of a Chi matrix.

    Returns:
        np.ndarray: a quantum process tomography Chi matrix. The Qiskit link
        above is a good description of a Chi matrix.
        Dict[str, float]: a dict of Pauli errors where the key
        is the type of Pauli error as a string (e.g., 'I', 'X', 'XI', 'XZX')
        and the value is the probability of this Pauli error.
    """
    n_qubits = int(np.round(np.log(chi.shape[0]) / np.log(4)))
    out = {}
    chi_diag = np.diag(chi)
    indices = np.nonzero(chi_diag)[0]
    for idx in indices:
        out[index_to_pauli_label(n_qubits=n_qubits, index=idx)] = chi_diag[idx]
    return out


def pauli_label_to_index(pauli_str: str) -> int:
    label_to_int = dict(zip('IXYZ', range(4)))
    sum = 0
    for i, c in enumerate(pauli_str[::-1]):
        try:
            sum += label_to_int[c] * (4**i)
        except KeyError as e:
            raise ValueError(
                f'Unrecognized Pauli error label "{pauli_str}"'
            ) from e
    return sum


def index_to_pauli_label(n_qubits: int, index: int) -> str:
    paulis = 'IXYZ'
    label = ''
    for i in reversed(range(n_qubits)):
        factor = 4**i
        term = index // factor
        label += paulis[term]
        index -= term * factor
    return label


def full_flip_error(
    linearized_probs: Union[np.ndarray, List[float], List[List[float]]]
) -> np.ndarray:
    """The general formula for the probabilities of Pauli errors from their
    linearized versions in the region close to 0.

    Assume pX ~ rX * t, pY ~ rY * t, and pZ ~ rZ * t when t close to 0.

    Since Pauli errors combine in the following ways: X then Y is equivalent to
    Z, X then Z to Y, etc,
    it follows that we can build a system of first-order linear differential
    equations relating pX(t), pX'(t), pY(t), pY'(t), pZ(t), pZ'(t).

    Solving it, we get the following formula for pX:
    pX(t) = 0.25 * (1 + exp(- 2 * (rY + rZ) * t)
                      - exp(- 2 * (rX + rZ) * t)
                      - exp(- 2 * (rX + rY) * t))
    and symmetric formulas for pY and pZ.

    This function implements these formulas."""
    linearized_probs = np.atleast_2d(linearized_probs)
    exps = np.exp(
        -2
        * (
            np.roll(linearized_probs, 1, axis=1)
            + np.roll(linearized_probs, 2, axis=1)
        )
    )
    return np.clip(
        0.25
        * np.squeeze(
            1 + exps - np.roll(exps, 1, axis=1) - np.roll(exps, 2, axis=1)
        ),
        0,
        1,
    )


def compose_1q_errors(
    a: Dict[str, float], b: Dict[str, float]
) -> Dict[str, float]:
    """Compose two sing-qubit Pauli error channels"""
    a_diag = np.zeros((4,)) if len(a) == 0 else pauli_errors_to_chi_diag(a)
    b_diag = np.zeros((4,)) if len(b) == 0 else pauli_errors_to_chi_diag(b)
    a_v = a_diag[1:]
    b_v = b_diag[1:]
    output_v = (
        a_v * b_diag[0]
        + b_v * a_diag[0]
        + np.roll(a_v, 1) * np.roll(b_v, 2)
        + np.roll(a_v, 2) * np.roll(b_v, 1)
    )
    return {
        'X': output_v[0],
        'Y': output_v[1],
        'Z': output_v[2],
    }


def tensor_errors(
    a: Dict[str, float], b: Dict[str, float]
) -> Dict[str, float]:
    """Build a Pauli error channel that is the tensor product of two
    lower-dimension Pauli error channels.

    For example, if the first Pauli channel is (I, X) and the second one is
    (II, YZ), then the output is (III, YZI, IIX, YZX)."""
    id_a = 1.0 - sum(a.values())
    id_b = 1.0 - sum(b.values())
    id_str_a = len(next(iter(a))) * 'I'
    id_str_b = len(next(iter(b))) * 'I'
    output = {}
    for b_pauli, b_prob in b.items():
        output[b_pauli + id_str_a] = id_a * b_prob
    for a_pauli, a_prob in a.items():
        output[id_str_b + a_pauli] = id_b * a_prob
    for (a_pauli, a_prob), (b_pauli, b_prob) in product(a.items(), b.items()):
        output[b_pauli + a_pauli] = a_prob * b_prob
    return output


def get_init_params(cls: Type) -> List[str]:
    """Get the names of the parameters of a class's __init__ method"""
    init_signature = signature(cls.__init__)
    return list(init_signature.parameters.keys())[1:]  # Skip 'self'
