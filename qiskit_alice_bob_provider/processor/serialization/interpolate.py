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

from enum import Enum
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate

from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
)

from ..description import ProcessorDescription
from .model import SerializedInstruction, SerializedProcessor

# A function that is the output of an interpolation method.
# In this context, interpolation is used to obtain estimates of quantum
# errors and instruction durations for parameter values that were not in the
# generated simulation data.
# For instance, we may have simulated errors for the values 3 and 4 of nbar.
# This interpolation function will be able to generate an error estimate for
# nbar=3.5.
Interpolator = Callable[[Union[float, List[float], np.ndarray]], np.ndarray]


class InterpolatedProcessor(ProcessorDescription):
    """A type of processor description built from a serialized representation
    of the processor behavior.
    """

    def __init__(
        self,
        serialized_processor: SerializedProcessor,
        clock_cycle: float = 1e-9,
    ) -> None:
        self.clock_cycle = clock_cycle
        self._instructions: Dict[
            Tuple[str, Tuple[int, ...]], SerializedInstruction
        ] = {}
        self._pauli_interp: Dict[
            Tuple[str, Tuple[int, ...]], Optional[Interpolator]
        ] = {}
        self._duration_interp: Dict[
            Tuple[str, Tuple[int, ...]], Interpolator
        ] = {}
        for instr in serialized_processor.instructions:
            pauli_interp = _build_interpolator(instr, _InterpolatedField.PAULI)
            duration_interp = _build_interpolator(
                instr, _InterpolatedField.DURATION
            )
            assert duration_interp is not None
            for qubits in instr.qubits:
                key = (instr.name, tuple(qubits))
                if key in self._instructions:
                    raise ValueError(
                        'An instruction already exist in the serialized'
                        ' processor model for the combination (instruction ='
                        f' {instr.name}, qubits = {qubits})'
                    )
                self._instructions[key] = instr
                self._pauli_interp[key] = pauli_interp
                self._duration_interp[key] = duration_interp

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for (name, qubits), instr in self._instructions.items():
            yield InstructionProperties(
                name=name,
                qubits=qubits,
                params=instr.free_params,
                readout_errors=instr.readout_errors,
            )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        pauli_interp = self._pauli_interp[(name, qubits)]
        duration_interp = self._duration_interp[(name, qubits)]
        instr = self._instructions[(name, qubits)]
        quantum = (
            None if pauli_interp is None else np.diag(pauli_interp(params))
        )
        return AppliedInstruction(
            duration=float(duration_interp(params)[0]),
            quantum_errors=quantum,
            readout_errors=instr.readout_errors,
        )


class InterpolationError(Exception):
    """An error happening when the user requested a parameter (rotation angle,
    gate duration, nbar) that is outside the range provided by the noise model.
    """


class _InterpolatedField(Enum):
    PAULI = 'PAULI'
    DURATION = 'DURATION'


def _build_interpolator(
    instruction: SerializedInstruction, field: _InterpolatedField
) -> Optional[Interpolator]:
    """From the quantum errors of an instruction, build an interpolator
    function able to estimate Pauli error probabilities and the instruction
    duration at parameter values not present in the input data.

    This interpolator will raise an error if interpolation was impossible. This
    happens if the requested point is outside the convex hull."""
    if len(instruction.free_params) == 0:
        point = instruction.interpolation_points[0]
        constant = (
            point.pauli_probabilities
            if field is _InterpolatedField.PAULI
            else [point.duration]
        )
        if constant is None:
            return None

        def constant_interpolator(
            # Can't use "_" because this would cause mypy to not understand
            # that this function is an Interpolator.
            # pylint: disable=unused-argument
            x: Union[float, List[float], np.ndarray]
        ) -> np.ndarray:
            return np.array(constant)

        return constant_interpolator

    params, values = _build_interpolation_matrices(instruction, field)
    if params.shape[0] == 0:
        return None

    # When params are of very different magnitudes (e.g., nbar between 1 and
    # 100 and a gate duration around 1e-7), the interpolation is skewed.
    # The rescaling below mitigates this effect.
    mean = np.mean(params, axis=0)
    std = np.std(params, axis=0)
    # The next line is not really useful since scipy will refuse to build
    # an interpolator anyway if one coordinate has a constant value.
    std[std == 0] = 1
    rescaled_params = (params - mean) / std

    if params.shape[1] == 1:
        scipy_interpolator = interpolate.interp1d(
            np.squeeze(rescaled_params), values, axis=0
        )
    else:
        scipy_interpolator = interpolate.LinearNDInterpolator(
            rescaled_params, values
        )

    def protected_interpolator(
        x: Union[float, List[float], np.ndarray]
    ) -> np.ndarray:
        error_message = (
            f'Could not interpolate requested point ({x}) because it is '
            f'out of the convex hull (instruction "{instruction.name}")'
        )
        try:
            y = scipy_interpolator((x - mean) / std)
        except ValueError as e:
            raise InterpolationError(error_message) from e
        if np.isnan(np.sum(y)):
            raise InterpolationError(error_message)
        return np.atleast_1d(np.squeeze(y))

    return protected_interpolator


def _build_interpolation_matrices(
    instruction: SerializedInstruction, field: _InterpolatedField
) -> Tuple[np.ndarray, np.ndarray]:
    """From the interpolation data points of a processor instruction, build
    matrices of the appropriate shape (Pauli error probs + duration) for use
    with scipy.interpolate functions."""
    x, y = [], []
    for point in instruction.interpolation_points:
        x_row = []
        for param in instruction.free_params:
            x_row.append(point.params[param])
        value = (
            point.pauli_probabilities
            if field is _InterpolatedField.PAULI
            else [point.duration]
        )
        if value is not None:
            x.append(x_row)
            y.append(value)
    return (np.array(x), np.array(y))
