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

from itertools import product
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)
from .utils import full_flip_error, pauli_errors_to_chi


class PhysicalCatProcessor(ProcessorDescription):
    """A description of a quantum processor made of physical cat qubits.

    All cat qubits are assumed to have the same physical properties, entirely
    controlled by the following quantities:
    * kappa_1 in Hz, the one-photon dissipation rate of the memory
    * kappa_2 in Hz, the two-photon dissipation rate of the memory
    * alpha (unitless), the amplitude of the cat state or coherent state in the
      memory. The average number of photons in the memory is equal to
      |alpha|^2.

    Note that on an actual chip, kappa_1 and kappa_2 depend on the architecture
    and cannot be changed. The amplitude alpha however can be adjusted by the
    operator of the chip.

    The formulas for gate times and error models used in this description are
    taken from different sources referenced in the code below.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_qubits: int = 5,
        kappa_1: float = 100,
        kappa_2: float = 10_000_000,
        average_nb_photons: float = 16,
        clock_cycle: float = 1e-9,
        coupling_map: Optional[List[Tuple[int, int]]] = None,
    ):
        self._n_qubits = n_qubits
        if average_nb_photons < 4.0:
            raise ValueError(
                'The average number of photons should be a float >= 4.0'
            )
        if kappa_1 < 10:
            raise ValueError(
                'The one-photon dissipation rate kappa_1 (Hz) should be a'
                ' float >= 10'
            )
        if kappa_1 / kappa_2 < 1e-7 or kappa_1 / kappa_2 > 1e-1:
            raise ValueError(
                'The ratio kappa_1 / kappa_2 should be between 1e-7 and 1e-1'
            )
        self._kappa_1 = kappa_1
        self._kappa_2 = kappa_2
        self._average_nb_photons = average_nb_photons
        self.clock_cycle = clock_cycle
        if coupling_map is None:
            # All-to-all coupling map
            coupling_map = list(
                (a, b)
                for a, b in product(range(n_qubits), range(n_qubits))
                if a != b
            )
        else:
            # A basic check of the validity of the coupling map
            for a, b in coupling_map:
                if a < 0 or b < 0 or a >= n_qubits or b >= n_qubits or a == b:
                    raise ValueError(
                        f'Coupling map contains an invalid pair ({a}, {b})'
                        f' for a processor with {n_qubits} qubits.'
                    )
        self._coupling_map = coupling_map

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for i in range(self._n_qubits):
            yield InstructionProperties(
                name='delay', params=['duration'], qubits=(i,)
            )
            yield InstructionProperties(
                name='rz', params=['angle'], qubits=(i,)
            )
            for inst in ['x', 'z', 'p0', 'p1', 'p+', 'p-', 'mx', 'mz']:
                yield InstructionProperties(name=inst, params=[], qubits=(i,))
        for i, j in self._coupling_map:
            yield InstructionProperties(name='cx', params=[], qubits=(i, j))

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        if name == 'mx':
            duration, errors = _mx_error(
                k1=self._kappa_1,
                k2=self._kappa_2,
                nbar=self._average_nb_photons,
            )
        elif name == 'mz':
            duration, errors = _mz_error(nbar=self._average_nb_photons)
        elif name == 'delay':
            assert len(params) == 1
            duration = params[0]
            errors = _idle_error(
                k1=self._kappa_1, nbar=self._average_nb_photons, t=duration
            )
        elif name in ['p+', 'p-']:
            duration, errors = _prep_plus_error(
                k1=self._kappa_1,
                k2=self._kappa_2,
                nbar=self._average_nb_photons,
            )
        elif name in ['p0', 'p1']:
            duration, errors = _prep_0_error(
                k2=self._kappa_2, nbar=self._average_nb_photons
            )
        elif name == 'x':
            duration, errors = _x_error(
                k1=self._kappa_1,
                k2=self._kappa_2,
                nbar=self._average_nb_photons,
            )
        elif name == 'rz':
            assert len(params) == 1
            duration, errors = _rz_error(
                k1=self._kappa_1,
                k2=self._kappa_2,
                nbar=self._average_nb_photons,
                theta=params[0],
            )
        elif name == 'z':
            duration, errors = _rz_error(
                k1=self._kappa_1,
                k2=self._kappa_2,
                nbar=self._average_nb_photons,
                theta=np.pi,
            )
        elif name == 'cx':
            duration, errors = _cx_error(
                k1=self._kappa_1,
                k2=self._kappa_2,
                nbar=self._average_nb_photons,
            )
        else:
            raise ValueError(f'Unknown instruction name "{name}"')
        try:
            quantum_errors = pauli_errors_to_chi(errors)
        except ValueError as e:
            raise ValueError(
                'The parameters of the processor (average_nb_photons='
                f'{self._average_nb_photons}, kappa_1={self._kappa_1}, '
                f'kappa_2={self._kappa_2}) led to inconsistent error '
                f'probabilities for instruction "{name}"'
            ) from e
        return AppliedInstruction(
            duration=duration,
            quantum_errors=quantum_errors,
            readout_errors=None,
        )


def _idle_error(k1, nbar, t):
    # [LES-HOUCHES] https://arxiv.org/pdf/2203.03222.pdf
    # The prefactor 1.1e-3 was chosen to match the alpha**2=8 point of the blue
    # curve in Fig. 7, p. 29:
    # The total bitflip probability (px+py) must be 1e-11 for alpha**2=8,
    # k1/k2=1e-2, t=1/k2.
    bit_flip = 0.5 * 1.1e-3 * nbar * k1 * np.exp(-2 * nbar) * t
    phase_flip = k1 * nbar * t
    x, y, z = full_flip_error([bit_flip, bit_flip, phase_flip])
    return {
        'X': x,
        'Y': y,
        'Z': z,
    }


def _prep_plus_error(k1, k2, nbar):
    # [AB-SHOR] https://arxiv.org/pdf/2302.06639v1.pdf
    # Page 25
    t = 1.0 / k2
    errors = {'Z': nbar * k1 * t}
    return t, errors


def _prep_0_error(k2, nbar):
    # [AWS-2022] https://arxiv.org/pdf/2012.04108.pdf
    # Table II, p. 17
    t = 0.1 / k2 / nbar
    errors = {'X': 0.39 * np.exp(-4 * nbar)}
    return t, errors


def _x_error(k1, k2, nbar):
    t = 1.0 / k2
    return t, _idle_error(k1, nbar, t)


def _mx_error(k1, k2, nbar):
    # [AB-SHOR] https://arxiv.org/pdf/2302.06639v1.pdf
    # Page 25
    t = 1.0 / k2
    errors = {
        'Z': nbar * k1 * t,
    }
    return t, errors


def _mz_error(nbar):
    # [AWS-2022] https://arxiv.org/pdf/2012.04108.pdf
    # Eq. 38, p. 18
    t = 850e-9
    errors = {
        'X': np.exp(-1.5 - 0.9 * nbar),
    }
    return t, errors


def _rz_error(k1, k2, nbar, theta):
    # [JEREMIE] https://hal.science/tel-03509305/document
    # p. 65
    # (Careful, there is an error in the formula: it should be |theta| instead
    # of sqrt(theta).)
    alpha = np.sqrt(nbar)
    t = 0.25 * np.abs(theta) / (alpha**3 * np.sqrt(k1 * k2))
    x, y, z = full_flip_error(
        [
            0,
            0,
            np.abs(theta) / (2 * alpha) * np.sqrt(k1 / k2),
        ]
    )
    errors = {
        'X': x,
        'Y': y,
        'Z': z,
    }
    return t, errors


def _cx_error(k1, k2, nbar):
    # [AB-SHOR] https://arxiv.org/pdf/2302.06639v1.pdf
    # Page 25
    t = 1 / k2
    zi_error = nbar * k1 * t + np.pi**2 / 64 / nbar / k2 / t
    zz_error = 0.5 * nbar * k1 * t

    # The prefactor 2631 was chosen so that the total probability of bit flip
    # is equal to 0.5exp(-2alpha**2) with alpha**2=19 and k1/k2=1e-5.
    # This is to match Eq. D8, p. 26.
    xi_error = 2631 * nbar * k1 * np.exp(-2 * nbar) * t / 6

    # This is acceptable because this error is 100x smaller than XI.
    iy_error = 0

    errors = {
        'IZ': zi_error,
        'ZZ': zz_error,
        'ZI': zz_error,
        'IX': xi_error,
        'XX': xi_error,
        'XI': xi_error,
        'IY': xi_error,
        'XY': xi_error,
        'XZ': xi_error,
        'YI': iy_error,
        'YY': iy_error,
        'YX': iy_error,
        'ZX': iy_error,
        'ZY': iy_error,
        'YZ': iy_error,
    }
    return t, errors
