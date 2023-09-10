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

from typing import Dict, List, Optional

from pydantic import BaseModel


class ProcessorMetadata(BaseModel):
    """Information about the processor model for debugging purposes"""

    schema_version: str
    generated_at: str
    name: str


class InterpolationPoint(BaseModel):
    """A description of the quantum noise and the instruction duration when
    the instruction is applied with a given set of parameters.

    Note that this representation can only describe Pauli errors, not general
    quantum process tomography Chi matrices."""

    # The probabilities of Pauli errors (the diagonal of a quantum process
    # tomography Chi matrix). The values should sum up to one.
    # Indices should with the first qubit to the right.
    # Example for one qubit: I, X, Y, Z
    # Example for two qubits: II, IX, IY, IZ, XI, ...
    pauli_probabilities: Optional[List[float]] = None

    # The duration of the instruction
    duration: float

    # The parameters for which the above Pauli error probabilities and duration
    # are valid. Example parameters: nbar, gate_duration_ns (for delay), angle
    # (for a rotation).
    params: Dict[str, float]


class SerializedInstruction(BaseModel):
    """The serialized representation of a given instruction"""

    # The name of the instruction: x, z, rz, mx, mz, delay, p0, p+, ...
    name: str

    # The indices of qubits this noise model is applicable to.
    # To support multi-qubit instructions like cx (CNOT), this is a list of
    # lists.
    # Example: noise model for x gate on qubits 1, 3, 5: [[1], [3], [5]]
    # Example: noise model for cx gate on qubits 1-2, 3-4: [[1, 2], [3, 4]]
    qubits: List[List[int]]

    # The parameters for which this noise model provides a grid of values.
    # The parameters names in the InterpolationPoint.params dictionary must
    # also appear in free_params.
    # Example for a rotation around the y-basis: ['angle', 'nbar']
    # Example for a delay instruction: ['gate_duration_ns', 'nbar']
    # Example for a x gate: ['nbar']
    free_params: List[str]

    # The list of parameters that were used in the simulation that generated
    # this noise model. This dictionary is here for debugging purposes only.
    fixed_params: Dict[str, float]

    # A list of objects representing the quantum noise and the instruction
    # duration when the instruction is applied with a given set of parameters.
    interpolation_points: List[InterpolationPoint]

    # Probability of reading an incorrect value when measuring the state of
    # a qubit. Only measurement instructions (mz and mx) should contain the
    # readout attribute.
    # It is assumed that the readout errors are independent of the free params.
    # Format: [P(1|0), P(0|1)].
    # Example: ideal = no readout noise = [0.0, 0.0]
    readout_errors: Optional[List[float]] = None


class SerializedProcessor(BaseModel):
    """A representation of the properties of a processor serialized in a static
    file.

    In practice, the instruction duration, quantum errors, and readout errors
    will be interpolated between the data points given by the serialized
    representation."""

    metadata: ProcessorMetadata
    instructions: List[SerializedInstruction]
