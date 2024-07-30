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

from typing import Dict, List

from qiskit.circuit import Delay, Instruction, Measure, Parameter
from qiskit.circuit.library import Initialize
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping,
)

from ..custom_instructions import MeasureX
from ..processor.description import InstructionProperties


def processor_to_qiskit_instruction(
    proc_instruction: InstructionProperties,
) -> Instruction:
    """Convert a procession instruction into its Qiskit implementation.

    If a new processor instruction cannot be straightforwardly mapped to a
    Qiskit instruction, it must be explicitly added in this function."""
    params = (Parameter(p) for p in proc_instruction.params)
    if proc_instruction.name == 'delay':
        return Delay(*params)
    elif proc_instruction.name in _known_preparations:
        return Initialize(proc_instruction.name[1], *params)
    elif proc_instruction.name in _known_measurements:
        return _known_measurements[proc_instruction.name](*params)
    elif proc_instruction.name in _known_rotations:
        return _known_rotations[proc_instruction.name](*params)
    elif proc_instruction.name in _known_no_arg_unitaries:
        return _known_no_arg_unitaries[proc_instruction.name](*params)
    else:
        raise NotImplementedError(
            f'Unsupported "{proc_instruction.name}" processor instruction'
            ' found, cannot convert to a Qiskit instruction'
        )


_standard_gates = get_standard_gate_name_mapping()
_known_no_arg_unitaries: Dict[str, type] = {
    name: inst.__class__
    for name, inst in _standard_gates.items()
    if len(inst.params) == 0
}
_known_rotations: Dict[str, type] = {
    name: _standard_gates[name].__class__ for name in ['rx', 'ry', 'rz']
}
_known_preparations: List[str] = ['p0', 'p+', 'p1', 'p-']
_known_measurements: Dict[str, type] = {
    'mz': Measure,
    'mx': MeasureX,
}
