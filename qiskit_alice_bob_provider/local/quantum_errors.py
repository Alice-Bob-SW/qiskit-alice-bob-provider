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

from typing import Callable, Dict, List, Optional, Sequence

from qiskit.circuit import Delay, Instruction, QuantumCircuit
from qiskit.circuit.equivalence_library import (
    EquivalenceLibrary,
    SessionEquivalenceLibrary,
)
from qiskit.circuit.library import IGate, Initialize
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Chi
from qiskit.transpiler import TransformationPass
from qiskit_aer.noise import pauli_error

from ..processor.description import InstructionProperties, ProcessorDescription
from ..processor.utils import chi_to_pauli_errors, is_diagonal
from .patch.local_noise_pass import LocalNoisePass
from .proc_to_qiskit import processor_to_qiskit_instruction


def build_quantum_error_passes(
    processor: ProcessorDescription,
) -> List[TransformationPass]:
    """From the description of a processor, build transpilation passes
    that insert quantum noise instructions representing the imperfection of the
    quantum processor.

    Note that the readout errors are not inserted by these transpilation
    passes. Readout errors are handled separately and are contined in the
    simulation NoiseModel.
    """

    if getattr(processor, 'noiseless', False):
        # By definition, a noiseless cat doesn't produce any noise,
        # therefore, we return an empty list of transformation passes.
        return []

    all_to_all = processor.all_to_all_connectivity

    passes = [_AddMeasureMarkerPass()]
    for instruction in processor.all_instructions():
        if all_to_all and instruction.qubits is not None:
            raise ValueError(
                'In a processor with all-to-all connectivity, instructions '
                "can't be attached to qubits. "
                f'Faulty instruction: {instruction}'
            )
        if not all_to_all and instruction.qubits is None:
            raise ValueError(
                'In a processor without all-to-all connectivity, instructions '
                'must be attached to specific qubits. '
                f'Faulty instruction: {instruction}'
            )
        pass_ = _transpilation_pass_from_instruction(processor, instruction)
        if pass_ is not None:
            passes.append(pass_)

    return passes


class _MeasureMarkerGate(IGate):
    """A virtual gate that is inserted before every measure instruction.

    Since a LocalNoisePass cannot act on a measure instruction, we insert
    marker gates before measure instructions, and then insert the measure
    instruction's quantum error after the marker gate. This is a trick that we
    may not need in the future, depending on the evolution of Qiskit."""

    def __init__(self, label: Optional[str] = None):
        super().__init__(label=label)

    def add_equivalence(self, library: EquivalenceLibrary) -> None:
        """Add an implementation of this marker gate to an equivalence library
        so that qiskit_aer can interpret the marker gate"""
        circuit = QuantumCircuit(1, 0)
        circuit.id(0)
        library.add_equivalence(self, circuit)


# pylint: disable=too-many-ancestors
class _MxMarkerGate(_MeasureMarkerGate):
    """A virtual gate that is inserted before every MeasureX"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(label=label)


# pylint: disable=too-many-ancestors
class _MzMarkerGate(_MeasureMarkerGate):
    """A virtual gate that is inserted before every MeasureZ"""

    def __init__(self, label: Optional[str] = None):
        super().__init__(label=label)


_marker_gate_types: Dict[str, type] = {
    'measure': _MzMarkerGate,
    'measure_x': _MxMarkerGate,
}


# These equivalences are added so that qiskit_aer can interpret the marker
# gates.
for marker_gate_type in _marker_gate_types.values():
    marker_gate_type().add_equivalence(SessionEquivalenceLibrary)


class _AddMeasureMarkerPass(TransformationPass):
    """A transpilation pass that inserts a virtual marker gate before every
    measure instruction.

    Since a LocalNoisePass cannot act on a measure instruction, we insert
    marker gates before measure instructions, and then insert the measure
    instruction's quantum error after the marker gate. This is a trick that we
    may not need in the future, depending on the evolution of Qiskit."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        marked_instructions = set(_marker_gate_types.keys())
        for node in dag.topological_op_nodes():
            if node.name not in marked_instructions:
                continue
            new_dag = DAGCircuit()
            new_dag.add_qubits(node.qargs)
            new_dag.add_clbits(node.cargs)
            marker_class = _marker_gate_types[node.name]
            new_dag.apply_operation_back(marker_class(), qargs=node.qargs)
            new_dag.apply_operation_back(
                node.op, qargs=node.qargs, cargs=node.cargs
            )
            dag.substitute_node_with_dag(node, new_dag)
        return dag


# The signature of the kind of functions accepted by qiskit_aer's
# LocalNoisePass.
_Pass = Callable[[Instruction, Sequence[int]], Optional[Instruction]]


def _transpilation_pass_from_instruction(
    processor: ProcessorDescription,
    instruction: InstructionProperties,
) -> LocalNoisePass:
    """From an instruction defined in the processor, create a
    LocalNoisePass inserting quantum noise after occurrences of the
    instruction in question."""

    qiskit_instruction = processor_to_qiskit_instruction(instruction)
    pass_factory = _pass_factory(
        processor=processor,
        instr_properties=instruction,
    )
    op_types = [
        _marker_gate_types[qiskit_instruction.name]
        if qiskit_instruction.name in _marker_gate_types
        else qiskit_instruction.base_class
    ]

    return LocalNoisePass(
        pass_factory,
        op_types=op_types,
        method='append',
    )


def _pass_factory(
    processor: ProcessorDescription, instr_properties: InstructionProperties
) -> _Pass:
    """Build a Qiskit transpiler pass function for a given processor
    instruction."""

    # Parameters need to be adapted to the processor params format.
    # Some instruction types must have their params handled in a specific way
    # (e.g., Delay, Initialize).
    param_handler_func = _param_handler_factory(processor, instr_properties)

    def _pass(
        instruction: Instruction, qubits: Sequence[int]
    ) -> Optional[Instruction]:
        qubits_ = tuple(qubits)
        if (
            instr_properties.qubits is not None
            and qubits_ != instr_properties.qubits
        ):
            # if the instruction is not all-to-all and the qubits don't match,
            # insert nothing in circuit
            return None

        params = param_handler_func(instruction)
        if params is None:
            # if params don't match (e.g., Initialize(+) when we're testing
            # for Initialize(0)), insert nothing in circuit
            return None

        chi_matrix = processor.apply_instruction(
            name=instr_properties.name,
            qubits=qubits_,
            params=params,
        ).quantum_errors
        if chi_matrix is None:
            # if there is no quantum noise for this instruction, insert nothing
            # in circuit
            return None
        if is_diagonal(chi_matrix) and instruction.condition is None:
            # if the matrix is diagonal, insert the noise as Pauli errors
            # rather than as a Chi/Kraus instruction.
            # This allows the user to use a Clifford simulator like
            # "stabilizer".
            # About "instruction.condition is None":
            # Unfortunately, Qiskit aer's pauli errors and Qiskit's
            # conditional instruction c_if are incompatible: the condition is
            # not applied on a Pauli error. This is a Qiskit bug to investigate
            # someday.
            pauli_errors = chi_to_pauli_errors(chi_matrix)
            error_instr = pauli_error(
                list(pauli_errors.items())
            ).to_instruction()
        else:
            n_qubits = len(qubits_)
            # It is unclear why the 2**n_qubits is needed in the Qiskit
            # implementation of Chi matrices.
            error_instr = Chi(chi_matrix * (2**n_qubits)).to_instruction()

        # We wrap the error instruction into a sub-circuit so that we can give
        # it a human-friendly name.
        # This is useful for debugging purposes. This does not affect
        # simulation performance because the wrapping sub-circuit is transpiled
        # away later right before simulation.
        # Can't we just rename the error instruction itself? No because it
        # needs to have the name "kraus" for Qiskit Aer to understand what thi
        # is.
        wrapper_circ = QuantumCircuit(
            error_instr.num_qubits,
            error_instr.num_clbits,
            name=f'{instr_properties.name}_error',
        )
        wrapper_circ.append(
            error_instr, wrapper_circ.qubits, wrapper_circ.clbits
        )

        return wrapper_circ.to_instruction()

    return _pass


# A function that inspects an instruction in a Qiskit circuit and transforms
# its params to be compatible with the processor format.
# If the return value is None, it means the params do not match the
# requirements for the ongoing transpilation pass (e.g., reading Initialize(+)
# when testing for Initialize(0)).
_ParamHandler = Callable[[Instruction], Optional[List]]


def _param_handler_factory(
    processor: ProcessorDescription, instr_properties: InstructionProperties
) -> _ParamHandler:
    # For a given processor instruction, we compute a reference Qiskit
    # instruction. During the pass, we'll compare instructions in the circuit
    # with this reference instruction.
    reference = processor_to_qiskit_instruction(instr_properties)

    if isinstance(reference, Initialize):

        def _handle_initialize_params(
            requested: Instruction,
        ) -> Optional[List]:
            if (
                len(requested.params) != 1
                or requested.params[0] != reference.params[0]
            ):
                return None
            return []

        return _handle_initialize_params
    elif isinstance(reference, Delay):

        def _handle_delay_params(requested: Instruction) -> Optional[List]:
            multipliers = {
                's': 1,
                'ms': 1e-3,
                'us': 1e-6,
                'ns': 1e-9,
                'ps': 1e-12,
                'dt': processor.clock_cycle,
            }
            duration_s = requested.duration * multipliers[requested.unit]
            return [duration_s]

        return _handle_delay_params
    else:

        def _handle_regular_params(requested: Instruction) -> Optional[List]:
            return requested.params

        return _handle_regular_params
