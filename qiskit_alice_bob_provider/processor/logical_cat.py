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

from typing import Dict, Iterator, List, Tuple

import numpy as np

from .description import (
    AppliedInstruction,
    InstructionProperties,
    ProcessorDescription,
)
from .utils import (
    compose_1q_errors,
    full_flip_error,
    pauli_errors_to_chi,
    tensor_errors,
)

_1Q_INSTRUCTIONS = [
    'x',
    'z',
    'p0',
    'p1',
    'p+',
    'p-',
    'mx',
    'mz',
    't',
    'tdg',
    'h',
    's',
    'sdg',
]

NOISELESS_VALUES = {
    'distance': 15,
    'kappa_1': 100,
    'kappa_2': 10_000_000,
    'average_nb_photons': 19,
}

NOISELESS_CAT_ERROR = (
    'Cannot instantiate a noiseless LogicalCatProcessor with '
    'custom values for distance, kappa_1, kappa_2, or average_nb_photons'
)


class LogicalCatProcessor(ProcessorDescription):
    """A description of a logical quantum processor whose logical qubits are
    made of physical cat qubits.

    The physical cat qubits are assembled into logical qubits using a linear
    repetition code aiming to correct phase flips. Bit flips are already
    addressed by the physical error correction built into the physical cat
    qubits.

    All logical qubits are assumed to have the same properties,
    entirely controlled by the following quantities:
    * distance (unitless), the distance of the repetition code used to correct
      phase flips. This is also the number of physical cat qubits that form
      a logical qubit
    * kappa_1 in Hz, the one-photon dissipation rate of the memory of physical
      cat qubits
    * kappa_2 in Hz, the two-photon dissipation rate of the memory of physical
      cat qubits
    * alpha (unitless), the amplitude of the cat state or coherent state in the
      memory of physical cat qubits. The average number of photons in the
      memory is equal to |alpha|^2.

    Note that on an actual chip, kappa_1 and kappa_2 depend on the architecture
    and cannot be changed. The amplitude alpha however can be adjusted by the
    operator of the chip.

    Qubit connectivity is all-to-all.

    The formulas for gate times and error models used in this description are
    taken from different sources referenced in the code below.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        n_qubits: int = 5,
        distance: int = 11,
        kappa_1: float = 100,
        kappa_2: float = 10_000_000,
        average_nb_photons: float = 16,
        clock_cycle: float = 1e-9,
        noiseless: bool = False,
    ):
        self._validate_parameters(
            distance, kappa_1, kappa_2, average_nb_photons, noiseless
        )
        self._distance = distance
        self._kappa_1 = kappa_1
        self._kappa_2 = kappa_2
        self._average_nb_photons = average_nb_photons
        self.n_qubits = n_qubits
        self.clock_cycle = clock_cycle
        self.noiseless = noiseless

    @classmethod
    def create_noisy(
        cls,
        n_qubits: int = 5,
        distance: int = 11,
        kappa_1: float = 100,
        kappa_2: float = 10_000_000,
        average_nb_photons: float = 16,
        clock_cycle: float = 1e-9,
    ):
        return cls(
            n_qubits,
            distance,
            kappa_1,
            kappa_2,
            average_nb_photons,
            clock_cycle,
            noiseless=False,
        )

    @classmethod
    def create_noiseless(
        cls, n_qubits: int = 40, clock_cycle: float = 1e-9, **kwargs
    ):
        """
        Builds a noiseless instance of a LogicalCatProcessor.
        """
        if kwargs:
            raise ValueError(NOISELESS_CAT_ERROR)
        return cls(
            n_qubits=n_qubits,
            distance=NOISELESS_VALUES.get('distance', 15),
            kappa_1=NOISELESS_VALUES.get('kappa_1', 100),
            kappa_2=NOISELESS_VALUES.get('kappa_2', 100_000_000),
            average_nb_photons=NOISELESS_VALUES.get('average_nb_photons', 19),
            clock_cycle=clock_cycle,
            noiseless=True,
        )

    @staticmethod
    def _validate_parameters(
        distance: int,
        kappa_1: float,
        kappa_2: float,
        average_nb_photons: float,
        noiseless: bool,
    ) -> None:
        """
        Validate parameters for the processor.
        Raises:
            ValueError: If any parameter is invalid.
        """

        # If the user bypasses the
        # LogicalCatProcessor.from_noiseless constructor
        if noiseless:
            (
                _distance,
                _kappa_1,
                _kappa_2,
                _average_nb_photons,
            ) = NOISELESS_VALUES.values()
            if (
                distance != _distance
                or kappa_1 != _kappa_1
                or kappa_2 != _kappa_2
                or average_nb_photons != _average_nb_photons
            ):
                raise ValueError(NOISELESS_CAT_ERROR)

        if distance % 2 != 1 or distance < 3:
            raise ValueError(
                'The distance of the linear repetition code should be an odd '
                'number >= 3'
            )
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

    def all_instructions(self) -> Iterator[InstructionProperties]:
        yield InstructionProperties(
            name='delay', params=['duration'], qubits=None
        )
        for inst in _1Q_INSTRUCTIONS:
            yield InstructionProperties(name=inst, params=[], qubits=None)
        yield InstructionProperties(name='cx', params=[], qubits=None)
        yield InstructionProperties(name='ccx', params=[], qubits=None)

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        if name == 'delay':
            assert len(params) == 1
            duration = params[0]
            errors = (
                (
                    _idle_error(
                        d=self._distance,
                        k1=self._kappa_1,
                        k2=self._kappa_2,
                        nbar=self._average_nb_photons,
                        t=duration,
                    )
                )
                if not self.noiseless
                else None
            )
        elif name in _1Q_INSTRUCTIONS:
            duration = _discrete_gate_time(d=self._distance, k2=self._kappa_2)
            errors = (
                (
                    _1q_logical_error(
                        d=self._distance,
                        nbar=self._average_nb_photons,
                        k1=self._kappa_1,
                        k2=self._kappa_2,
                    )
                )
                if not self.noiseless
                else None
            )
        elif name == 'cx':
            duration = _discrete_gate_time(d=self._distance, k2=self._kappa_2)
            errors = (
                (
                    _cx_error(
                        d=self._distance,
                        k1=self._kappa_1,
                        k2=self._kappa_2,
                        nbar=self._average_nb_photons,
                    )
                )
                if not self.noiseless
                else None
            )
        elif name == 'ccx':
            duration = _discrete_gate_time(d=self._distance, k2=self._kappa_2)
            errors = (
                (
                    _ccx_error(
                        d=self._distance,
                        k1=self._kappa_1,
                        k2=self._kappa_2,
                        nbar=self._average_nb_photons,
                    )
                )
                if not self.noiseless
                else None
            )
        else:
            raise ValueError(f'Unknown instruction name "{name}"')
        try:
            # If the process is noiseless, the errors variable will be None
            # So we have to take account of that
            quantum_errors = pauli_errors_to_chi(errors) if errors else None
        except ValueError as e:
            raise ValueError(
                f'The parameters of the processor (distance={self._distance}, '
                f'average_nb_photons={self._average_nb_photons}, '
                f'kappa_1={self._kappa_1}, kappa_2={self._kappa_2}) led to '
                f'inconsistent error probabilities for instruction "{name}"'
            ) from e
        return AppliedInstruction(
            duration=duration,
            quantum_errors=quantum_errors,
            readout_errors=None,
        )


def _discrete_gate_time(d: int, k2: float) -> float:
    # [AB-SHOR] https://arxiv.org/pdf/2302.06639v1.pdf
    # p. 4.
    # The gate time is dominated by the duration of the error correction cycle.
    # The error correction cycle contains d measurement cycles, which take 5/k2
    # each.
    return 5 * d / k2


def _logical_bit_flip_error(d: int, nbar: float) -> float:
    # [AB-SHOR] https://arxiv.org/pdf/2302.06639v1.pdf
    # Eq. 3, p. 25.
    # This error is equal to d times the per measurement cycle error (at the
    # first order).
    return (d - 1) * d * np.exp(-2 * nbar)


def _logical_phase_flip_error(
    d: int, nbar: float, k1: float, k2: float
) -> float:
    # [AB-SHOR] https://arxiv.org/pdf/2302.06639v1.pdf
    # Eq. 3, p. 25.
    # This error is equal to d times the per measurement cycle error (at the
    # first order).
    return 5.6e-2 * d * (nbar**0.86 * k1 / k2 / 1.3e-2) ** (0.5 * (d + 1))


def _1q_logical_error(
    d: int, nbar: float, k1: float, k2: float
) -> Dict[str, float]:
    px = _logical_bit_flip_error(d=d, nbar=nbar)
    pz = _logical_phase_flip_error(d=d, nbar=nbar, k1=k1, k2=k2)
    x, y, z = full_flip_error([px, 0, pz])
    return {
        'X': x,
        'Y': y,
        'Z': z,
    }


def _idle_error(
    t: float, d: int, nbar: float, k1: float, k2: float
) -> Dict[str, float]:
    # Count the number of error correction cycles within duration t and compose
    # that many times the error correction cycle error.
    cycle_time = _discrete_gate_time(d=d, k2=k2)
    corr_cycles = int(t // cycle_time)
    cycle_error = _1q_logical_error(d=d, nbar=nbar, k1=k1, k2=k2)
    # Why not an empty dict for the initial error? If an empty dict, the Pauli
    # errors cannot be converted into a chi matrix (because the number of
    # qubits is undetermined)
    error: Dict[str, float] = {'X': 0, 'Z': 0}
    for _ in range(corr_cycles):  # this is under-optimized
        error = compose_1q_errors(error, cycle_error)
    return error


def _cx_error(d: int, nbar: float, k1: float, k2: float) -> Dict[str, float]:
    # The tensor product of two independent single-qubit errors
    single_error = _1q_logical_error(d=d, nbar=nbar, k1=k1, k2=k2)
    return tensor_errors(single_error, single_error)


def _ccx_error(d: int, nbar: float, k1: float, k2: float) -> Dict[str, float]:
    # The tensor product of three independent single-qubit errors
    single_error = _1q_logical_error(d=d, nbar=nbar, k1=k1, k2=k2)
    return tensor_errors(
        tensor_errors(single_error, single_error), single_error
    )
