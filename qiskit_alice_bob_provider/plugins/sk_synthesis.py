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
from inspect import isclass
from typing import Set

from qiskit.circuit import ControlFlowOp, Instruction, ParameterExpression
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping,
)
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import (
    AnalysisPass,
    PassManager,
    PassManagerConfig,
    Target,
    TranspilerError,
)
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from .state_preparation import StatePreparationPlugin


class UnboundParameterCheck(AnalysisPass):
    """Raises TranspilerError if an operation has an unbound parameter.

    The Solovay-Kitaev synthesis needs a numeric value for each rotation.
    """

    def run(self, dag: DAGCircuit) -> None:
        for node in dag.op_nodes():
            if isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    self.run(circuit_to_dag(block))
                continue
            names = sorted(
                {
                    parameter.name
                    for param in node.op.params
                    if isinstance(param, ParameterExpression)
                    for parameter in param.parameters
                }
            )
            if names:
                raise TranspilerError(
                    f'{node.op.name} depends on the unbound parameters '
                    f'{", ".join(names)}. Bind the parameters with '
                    'assign_parameters before transpile.'
                )


class SKSynthesisPlugin(PassManagerStagePlugin):
    """This plugin configures the Solavay-Kitaev synthesis for the special
    case of logical qubits made out of physical cat qubits.

    Here's what it does:
    * Compute the basis gates to be used for unitary synthesis. This is
      actually different from the basis gates supported by the backend: not
      all backend basis gates can be used as basis gates for the SK synthesis.
      Unfortunately, Qiskit does not allow setting a different basis gate
      set for the synthesis step, so we hack our way around it.
    * Restrict the SK synthesis to the 1-qubit gates of the SK basis gate
      set, with a recursion degree of 3 by default
    * Mix the SK synthesis with the transpilation passes from
      StatePreparationPlugin

    This plugin wouldn't exist if Qiskit's transpile exposed more options to
    configure the synthesis method (and didn't override the available options
    in strange ways)."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        # Compute the discrete basis gates for the Solovay-Kitaev synthesis
        target: Target = pass_manager_config.target
        discrete_basis_gates: Set[str] = set()
        discrete_1q_basis_gates: Set[str] = set()
        for instr, _ in target.instructions:
            # At this point, the control flow operations
            # are not yet instantiated which means the
            # instruction can be a class. We have to check if
            # the class is a subclass of 'ControlFlowOp',
            # in which case we can skip the rest of the loop
            # iteration to avoid throwing any errors.
            if isclass(instr) and issubclass(instr, ControlFlowOp):
                continue

            assert isinstance(instr, Instruction)
            if len(instr.params) != 0 or instr.name in {
                'measure',
                'measure_x',
                'reset',
                'delay',
                'cz',
                'ccz',
                'cswap',
            }:
                continue
            if instr.num_qubits == 1:
                discrete_1q_basis_gates.add(instr.name)
            discrete_basis_gates.add(instr.name)

        # Solovay-Kitaev synthesizes 1-qubit gates only. The translation
        # passes decompose the multi-qubit gates, and a second synthesis pass
        # discretizes the rotations that remain.
        synth_gates: Set[str] = set()
        for name, instr in get_standard_gate_name_mapping().items():
            if name in {
                'measure',
                'measure_x',
                'reset',
                'delay',
                'cz',
                'ccz',
                'cswap',
            }:
                continue
            if instr.num_qubits != 1:
                continue
            synth_gates.add(name)
        synth_gates -= set(discrete_basis_gates)

        # The default synthesis method is Solovay-Kitaev
        pass_manager_config.unitary_synthesis_method = (
            'sk'
            if pass_manager_config.unitary_synthesis_method == 'default'
            else pass_manager_config.unitary_synthesis_method
        )

        pass_manager_config.unitary_synthesis_plugin_config = {
            'depth': 5,
            'recursion_degree': 3,
            **(pass_manager_config.unitary_synthesis_plugin_config or {}),
        }

        # Use as basis the same pass manager as the LocalStatePreparationPlugin
        pm = StatePreparationPlugin().pass_manager(
            pass_manager_config=pass_manager_config,
            optimization_level=optimization_level,
        )

        # Update the UnitarySynthesis pass (which controls the
        # SolovayKitaevSynthesis plugin) with the gates to synthesize and basis
        # gates computed above.
        if pass_manager_config.unitary_synthesis_method == 'sk':
            for task in pm._tasks:
                for subtask in task:
                    if (
                        isinstance(subtask, UnitarySynthesis)
                        and subtask.method == 'sk'
                    ):
                        # There is no option to manually set the gates to
                        # synthesize in UnitarySynthesis, and the default
                        # _synth_gates is just 'unitary'!
                        subtask._synth_gates = synth_gates
                        # The Solovay-Kitaev decomposition accepts
                        # 1-qubit basis gates only.
                        subtask._basis_gates = discrete_1q_basis_gates

        return PassManager([UnboundParameterCheck()]) + pm
