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

from typing import Dict, List, Tuple

from qiskit.circuit import Instruction
from qiskit.transpiler import Target

from ..processor.description import ProcessorDescription
from .instruction_durations import ProcessorInstructionDurations
from .proc_to_qiskit import processor_to_qiskit_instruction


def processor_to_target(processor: ProcessorDescription) -> Target:
    """Create a Qiskit Target from a ProcessorDescription

    A ProcessorDescription is the description of the properties of a quantum
    processor. In this regard, a processor has the same function as a Qiskit
    Target.

    Please note that the Qiskit target produced by this function does not fully
    represent the array of operations available on the processor.

    This is because, contrary to what is advertised in the Target documentation
    at https://qiskit.org/documentation/stubs/qiskit.transpiler.Target.html,
    the Qiskit transpiler expects the names of instructions in a Target to be
    listed under their Qiskit operation names. Otherwise they won't be taken
    into account and used by some transpilation passes who have a basis_gates
    argument like UnrollCustomDefinitions.

    This is not a problem for scheduling because durations are obtained from
    the richer ProcessorInstructionDurations class which derives from
    InstructionDurations.
    This might be an issue for other types of transpilation passes.

    Here's a concrete example of this loss of information:

    If an Alice & Bob processor implements the following instructions:
    * p0, which translates to Initialize(0)
    * p+, which translates to Initialize (+)
    * rz, which translates to RzGate(Parameter('angle'))
    * rzpio2, which translates to RzGate(np.pi/2)

    Then the corresponding Qiskit Target will contain the following
    instructions:
    * 'initialize' --> Initialize(0)
    * 'rz' --> RzGate(Parameter('angle'))

    This loss of information cannot be avoided because Qiskit identifies
    instructions by their names for simplicity.

    The last instruction (rzpio2) is a made-up example for the sake of clarity.
    """
    target = Target(
        dt=processor.clock_cycle, num_qubits=processor.n_qubits or 0
    )

    # qiskit name -> supported qubit combinations
    qubits: Dict[str, List[Tuple[int, ...]]] = {}

    # Qiskit name -> a given Qiskit implementation of this instruction
    # As a consequence, if two proc instructions are implemented using the same
    # Qiskit implementation, they will conflict here. This is not an issue
    # though, because the Qiskit Target's instructions are only used for
    # gate basis conversion.
    example_instructions: Dict[str, Instruction] = {}

    for instruction in processor.all_instructions():
        qiskit_instr = processor_to_qiskit_instruction(instruction)
        name = qiskit_instr.name
        example_instructions[name] = qiskit_instr
        if name not in qubits:
            qubits[name] = []
        if instruction.qubits is not None:
            qubits[name].append(instruction.qubits)

    for name, qiskit_instr in example_instructions.items():
        properties = (
            None  # the instruction is all-to-all
            if len(qubits[name]) == 0
            # the instruction is attached to some qubits only
            else {qbts: None for qbts in qubits[name]}
        )
        target.add_instruction(
            instruction=qiskit_instr,
            properties=properties,
            name=qiskit_instr.name,
        )
    target._instruction_durations = (  # pylint: disable=protected-access
        ProcessorInstructionDurations(processor=processor)
    )
    return target
