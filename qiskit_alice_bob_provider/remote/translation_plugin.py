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

from typing import List, Optional, Union

import numpy as np
from qiskit.circuit import Instruction, Reset
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.transpiler import PassManager, TransformationPass
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from ..errors import AliceBobTranspilationException


class StatePreparationPass(TransformationPass):
    """
    A transpilation pass to manage the complex interaction between Qiskit's
    quantum_initalizer, qiskit-qir, and the Alice & Bob API.

    When the user calls `initialize` in their circuit, the Initialize
    instruction is ultimately transpiled away before conversion to QIR by
    qiskit-qir. Here's what happens:
    * '+' -> reset - h
    * '-' -> reset - x - h
    * '0' -> (results in an exception: remote_state_preparation not supported)
    * '1' -> reset - x
    * 0 -> (results in an exception: remote_state_preparation not supported)
    * 1 -> reset - x
    * [1, 0] -> (results in an exception: disentangler_dg not supported)
    * [0, 1] -> reset - ry(pi)
    * [1, 1] / sqrt(2) -> reset - ry(pi/2)
    * [1, -1] / sqrt(2) -> reset - ry(pi/2) - rz(pi)

    Besides the exceptions we need to fix, we'd also like state preparation to
    use only resets and rotations. This enables the Alice & Bob API to then
    transpile those rotations back to the initial user intent (prepare +,
    prepare -, etc). If state preparations used gates like x, then we would
    not be able to distinguish the sequence prepare(0) - x from prepare(1).

    This complex behavior and this transpilation pass will go away when
    qiskit-qir supports extending the language with custom mappings, like
    "qiskit's initialize(+) to qir's prepare_x(0)".
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if node.name != 'initialize':
                continue
            assert isinstance(node.op, Initialize)
            if len(node.qargs) > 1:
                raise AliceBobTranspilationException(
                    'Multi-qubit state preparation is not supported'
                )
            new_instruction = _substitute_single_qubit_initialize(
                node.op.params
            )
            if new_instruction is None:
                continue
            dag.substitute_node(node, new_instruction)
        return dag


class StatePreparationPlugin(PassManagerStagePlugin):
    def pass_manager(
        self,
        pass_manager_config,
        optimization_level=None,
    ) -> PassManager:
        return PassManager([StatePreparationPass()])


_State = Union[List[str], List[complex]]


# pylint: disable=too-many-return-statements,too-many-branches
def _substitute_single_qubit_initialize(
    state: _State,
) -> Optional[Instruction]:
    """For a given argument to the Initialize instruction, return an equivalent
    instruction that is either an Initialize instruction compatible with
    qiskit-qir or a reset.
    """
    if len(state) == 1 and isinstance(state[0], str):
        state_str = state[0]
        if state_str == '0':
            return Reset()
        elif state_str == '1':
            return Initialize([0, 1])
        elif state_str == '+':
            return Initialize([1 / np.sqrt(2), 1 / np.sqrt(2)])
        elif state_str == '-':
            return Initialize([1 / np.sqrt(2), -1 / np.sqrt(2)])
    elif len(state) == 1 and isinstance(state[0], complex):
        state_int = state[0]
        if state_int == 0:
            return Reset()
        elif state_int == 1:
            return Initialize([0, 1])
    elif len(state) == 2 and isinstance(state[0], complex):
        global_phase = np.angle(state[0])
        state_ph = np.array(state) * np.exp(-1j * global_phase)
        if np.allclose(state_ph, np.array([1, 0])):
            return Reset()
        elif np.allclose(state_ph, np.array([0, 1])):
            return Initialize([0, 1])
        elif np.allclose(state_ph, np.array([1, 1]) / np.sqrt(2)):
            return None
        elif np.allclose(state_ph, np.array([1, -1]) / np.sqrt(2)):
            return None
    raise AliceBobTranspilationException(
        f'The Initialize state preparation with desired state {state} is '
        'not supported'
    )
