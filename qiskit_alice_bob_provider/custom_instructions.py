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

from typing import Optional

from qiskit.circuit import Instruction, InstructionSet, QuantumCircuit, Reset
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.quantumcircuit import ClbitSpecifier, QubitSpecifier


class MeasureX(Instruction):
    """Quantum measurement in the X basis."""

    def __init__(self, label: Optional[str] = None):
        """Create a new X-measurement instruction."""
        super().__init__('measure_x', 1, 1, [], label=label)
        self._definition = QuantumCircuit(1, 1, name='measure_x')
        self._definition.h(0)
        self._definition.measure(0, 0)
        self._definition.h(0)


def _measure_x(
    self: QuantumCircuit, qubit: QubitSpecifier, cbit: ClbitSpecifier
) -> InstructionSet:
    return self.append(MeasureX(), [qubit], [cbit])


# Patching the QuantumCircuit class to add a `measure_x` method.
QuantumCircuit.measure_x = _measure_x

# Add MeasureX to the session equivalence library
_measure_x_inst = MeasureX()
SessionEquivalenceLibrary.add_equivalence(
    _measure_x_inst, _measure_x_inst.definition
)


# Add an equivalent from Reset to Initialize('0'). This is useful in the case
# of the local provider.
# Although we may want to preserve the Reset instruction for a future behavior
# not yet implemented (e.g., resetting to the void state of the cavity in
# case of cat qubits, which is not how the logical |0> state is encoded), users
# are used to use Reset to actually say Initialize('0').
# That's what this equivalence rule does. If it causes trouble in the future,
# it may be removed.
_c = QuantumCircuit(1, 0)
_c.initialize('0', 0)  # pylint: disable=no-member
SessionEquivalenceLibrary.add_equivalence(Reset(), _c)
