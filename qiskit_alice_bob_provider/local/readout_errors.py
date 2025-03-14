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

import warnings
from typing import List, Optional, Set, Tuple

from qiskit_aer.noise import NoiseModel, ReadoutError

from ..processor.description import ProcessorDescription


def build_readout_noise_model(
    proc: ProcessorDescription,
) -> Optional[NoiseModel]:
    """A noise model describing only readout errors: the errors happening
    during a measurement of the qubit state.

    All other errors ("quantum errors") are described as transpilation passes.
    See module `quantum_errors.py`.

    A readout error is the probability that the qubit is in state 0 and that
    the measurement device returns 1, or 1 -> 0, + ->- , - -> +, etc.

    Known limitation: Qiskit only supports one readout error per qubit. That's
    because Qiskit assumes there is only one type of measurement per qubit
    (measurement in the z-basis). For a cat, we can measure in the z and x
    basis, so we would need to assign two readout errors per qubit. This is
    impossible in the current state of Qiskit.
    """

    if getattr(proc, 'noiseless', False):
        # By definition, a noiseless cat doesn't produce any noise,
        # therefore, we return no NoiseModel.
        return None

    if proc.all_to_all_connectivity:
        return _build_all_to_all_readout_noise_model(proc)
    return _build_physical_readout_noise_model(proc)


def _build_physical_readout_noise_model(
    proc: ProcessorDescription,
) -> Optional[NoiseModel]:
    qiskit_noise_model = NoiseModel()
    qubits_with_readout_error: Set[Tuple[int, ...]] = set()

    for instr in proc.all_instructions():
        if instr.readout_errors is None:
            continue
        if instr.qubits is None:
            raise ValueError(
                'All-to-all instructions (i.e., not attached to specific '
                'qubits) are not supported in a physical processor. '
                f'Faulty instruction: {instr}'
            )
        error = _to_qiskit_readout_error(instr.readout_errors)
        if instr.qubits in qubits_with_readout_error:
            warnings.warn(
                UserWarning(
                    f'Qubit(s) {instr.qubits} already contain a readout error,'
                    ' cannot add another one. The readout error of instruction'
                    f' {instr.name} will be ignored.'
                )
            )
            continue
        qiskit_noise_model.add_readout_error(error=error, qubits=instr.qubits)
        qubits_with_readout_error.add(instr.qubits)

    return qiskit_noise_model


def _build_all_to_all_readout_noise_model(
    proc: ProcessorDescription,
) -> Optional[NoiseModel]:
    qiskit_noise_model = NoiseModel()
    has_error = False

    for instr in proc.all_instructions():
        if instr.readout_errors is None:
            continue
        if instr.qubits is not None:
            raise ValueError(
                'Instructions attached to specific qubits are not supported '
                'in an all-to-all processor. '
                f'Faulty instruction: {instr}'
            )
        if has_error:
            warnings.warn(
                UserWarning(
                    f'The all-to-all model already contain a readout error,'
                    ' cannot add another one. The readout error of instruction'
                    f' {instr.name} will be ignored.'
                )
            )
            continue
        error = _to_qiskit_readout_error(instr.readout_errors)
        qiskit_noise_model.add_all_qubit_readout_error(error=error)
        has_error = True

    return qiskit_noise_model


def _to_qiskit_readout_error(readout_errors: List[float]) -> ReadoutError:
    p_1_given_0, p_0_given_1 = readout_errors
    return ReadoutError(
        [[1 - p_1_given_0, p_1_given_0], [p_0_given_1, 1 - p_0_given_1]]
    )
