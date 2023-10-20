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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class InstructionProperties:
    """The description of an instruction available on the processor."""

    # The name and identifier of the instruction type
    name: str

    # The qubits on which this instruction can be applied.
    # The combination (name, qubits) completely identifies the instruction.
    # A value of None means the instructions can be applied on all tuples,
    # which is the case for all-to-all connectivity.
    qubits: Optional[Tuple[int, ...]]

    # The names of the parameters of the instruction
    params: List[str]

    # The readout assignment errors of a measurement instruction.
    # A value of [0.01, 0.02] is understood as P(read 1 | qubit in 0) = 0.01
    # and P(read 0 | qubit in 1) = 0.02.
    # Use this attribute when the readout errors are independent of the
    # instruction params. If param-dependent readout errors are found in
    # AppliedInstruction, they should take precedence.
    readout_errors: Optional[List[float]] = None


@dataclass
class AppliedInstruction:
    """The behavior of the instruction when applied with a given set of
    parameters."""

    # The duration of the instruction. Always in seconds.
    duration: float

    # The quantum noise of the instruction represented as a quantum process
    # tomography chi matrix. Refer to
    # https://qiskit.org/documentation/stubs/qiskit.quantum_info.Chi.html
    quantum_errors: Optional[np.ndarray]

    # The readout assignment errors of a measurement instruction.
    # A value of [0.01, 0.02] is understood as P(read 1 | qubit in 0) = 0.01
    # and P(read 0 | qubit in 1) = 0.02.
    # Use this attribute when the readout errors depend on the
    # instruction params. They should take precedence over param-independent
    # readout errors found in InstructionProperties.
    readout_errors: Optional[List[float]]


class ProcessorDescription(ABC):
    """An interface to represent the behavior of a quantum processor (QPU) at
    the gate level.

    This interface does not depend on Qiskit, and thus could be used to provide
    information about a QPU to another quantum computing framework.

    Additionally, this interface is finer-grained than a Qiskit Target, since
    it can represent instructions with noise and duration continuously
    dependent on the instruction parameters.
    """

    @abstractmethod
    def all_instructions(self) -> Iterator[InstructionProperties]:
        """Return all instructions available on the processor.

        If an instruction is available on multiple qubit combinations, it will
        be listed once per qubit combination.
        """

    @abstractmethod
    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        """For a given instruction, return the instruction duration and noise.

        If the instruction is not found (as represented by the combination of
        an instruction name and a tuple of qubits), the call will raise an
        exception.
        """

    # The clock cycle duration in the same time unit as other durations
    # in the ProcessorDescription (usually seconds).
    #
    # Note that the ProcessorDescription does not return durations in
    # multiples of the clock cyle and accepts durations (e.g., for delay)
    # that are not multiples of clock cycle.
    #
    # If respecting the clock cycle is important, it is up to the user (e.g.,
    # the Qiskit transpiler) to clip durations to clock cycle multiples.
    clock_cycle: float

    # The number of qubits accepted by the processor. This attribute should
    # only be set in a processor with all-to-all connectivity.
    # In a processor with instructions tied to tuples of qubits, the number of
    # qubits will be computed by Qiskit.
    n_qubits: Optional[int] = None

    @property
    def all_to_all_connectivity(self) -> bool:
        return self.n_qubits is not None
