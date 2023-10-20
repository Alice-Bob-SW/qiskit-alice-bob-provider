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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from qiskit.circuit import Barrier, Delay, Instruction, Parameter, Qubit
from qiskit.transpiler.instruction_durations import (
    InstructionDurations,
    InstructionDurationsType,
)

from ..processor.description import ProcessorDescription
from .proc_to_qiskit import processor_to_qiskit_instruction


class ProcessorInstructionDurations(InstructionDurations):
    """A more flexible version of InstructionDurations.

    InstructionDurations stores a duration for every supported combination of
    instruction (a name), qubits, and params. In this sense, it is some kind of
    fancy dict.

    This makes it impossible to represent a continuous relation between
    parameters and duration. For instance, InstructionDurations cannot
    represent a rotation around the Z-axis rz(theta) whose duration
    continuously depends on the angle theta.

    ProcessorInstructionDurations on the other hand accepts any parameter in
    :method:`ProcessorInstructionDurations.get` and forwards the requested
    parameters to the underlying ProcessorDuration.
    """

    def __init__(self, processor: ProcessorDescription):
        super().__init__(instruction_durations=None, dt=processor.clock_cycle)
        self._proc = processor
        self._qiskit_to_proc_mapping = _from_qiskit_to_proc_instructions(
            processor
        )

    def get(
        self,
        inst: Union[str, Instruction],
        qubits: Union[int, List[int], Qubit, List[Qubit]],
        unit: str = 'dt',
        parameters: Optional[List[float]] = None,
    ) -> float:
        """Get the duration of the instruction with the name, qubits, and
        parameters.

        Instructions may have a parameter-dependent durations

        Args:
            inst: An instruction or its name to be queried.
            qubits: Qubits or its indices that the instruction acts on.
            unit: The unit of duration to be returned. It must be 's' or 'dt'.
            parameters: The value of the parameters of the desired instruction.

        Returns:
            float: The duration of the instruction on the qubits.

        Raises:
            TranspilerError: No duration is defined for the instruction.
        """
        if isinstance(inst, Barrier):
            return 0
        elif isinstance(inst, Delay):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                converted = self._convert_unit(inst.duration, inst.unit, unit)
            if converted == 0:
                warnings.warn(
                    (
                        f'Duration is rounded to {converted} [dt] from'
                        f' {inst.duration} [{inst.unit}]'
                    ),
                    UserWarning,
                )
            return converted

        if isinstance(inst, Instruction):
            inst_name = inst.name
        else:
            inst_name = inst

        if isinstance(qubits, (int, Qubit)):
            qubits = [qubits]

        if isinstance(qubits[0], Qubit):
            # This explicit for loop pleases mypy (as opposed to a
            # for-comprehension).
            new_qubits = []
            for q in qubits:
                assert isinstance(q, Qubit)
                new_qubits.append(q.index)
            qubits = new_qubits

        if parameters is None and isinstance(inst, Instruction):
            parameters = inst.params

        with warnings.catch_warnings():
            # We expect some rounding when going from s to dt, so we simply
            # ignore those.
            warnings.simplefilter('ignore')
            return self._get(inst_name, qubits, unit, parameters)

    def _get(
        self,
        name: str,
        qubits: List[int],
        to_unit: str,
        parameters: Optional[Iterable[float]] = None,
    ) -> float:
        if name == 'barrier':
            return 0

        parameters = list(parameters) if parameters else []

        try:
            candidate_instructions = (
                # instruction tied to a tuple of qubit
                self._qiskit_to_proc_mapping.get((name, tuple(qubits)), None)
                # instruction with all-to-all instruction
                or self._qiskit_to_proc_mapping[(name, None)]
            )
        except KeyError as e:
            raise ValueError(
                _instruction_not_found(name, qubits, parameters)
            ) from e

        for instr, proc_instr_name in candidate_instructions:
            if _have_same_params(
                requested=parameters, definition=instr.params
            ):
                applied_instr = self._proc.apply_instruction(
                    name=proc_instr_name,
                    qubits=tuple(qubits),
                    params=parameters,
                )
                return self._convert_unit(applied_instr.duration, 's', to_unit)

        raise ValueError(_instruction_not_found(name, qubits, parameters))

    def update(
        self,
        inst_durations: Optional[InstructionDurationsType],
        dt: float = None,  # type: ignore
    ):
        """ProcessorInstructionDurations does not support updating its content
        like InstructionDurations. It is not a dict."""
        return NotImplementedError()


def _instruction_not_found(
    name: str, qubits: List[int], parameters: List[float]
) -> str:
    return (
        f'The processor does not support instruction "{name}" on qubits'
        f' {qubits} with params "{parameters}"'
    )


_InstructionDict = Dict[
    Tuple[str, Optional[Tuple[int, ...]]], List[Tuple[Instruction, str]]
]


def _from_qiskit_to_proc_instructions(
    processor: ProcessorDescription,
) -> _InstructionDict:
    """Create a dict mapping (Qiskit instruction name, qubits) to
    (Qiskit instruction, proc instruction name).
    """
    out: _InstructionDict = {}
    for instr in processor.all_instructions():
        qiskit_instr = processor_to_qiskit_instruction(instr)
        key = (str(qiskit_instr.name), instr.qubits)
        if key not in out:
            out[key] = []
        out[key].append((qiskit_instr, instr.name))
    return out


def _have_same_params(requested: List[Any], definition: List[Any]) -> bool:
    """Compare the parameters provided by the Qiskit instruction in the circuit
    with the parameters of a reference Qiskit instruction."""
    if len(requested) != len(definition):
        return False
    for req, def_ in zip(requested, definition):
        if isinstance(def_, Parameter):
            continue
        if req == def_:
            continue
        return False
    return True
