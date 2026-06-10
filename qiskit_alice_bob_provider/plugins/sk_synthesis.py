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
from functools import lru_cache
from inspect import isclass
from typing import FrozenSet, List, Set

from qiskit.circuit import ControlFlowOp, Instruction
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping,
)
from qiskit.synthesis import generate_basic_approximations
from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
    GateSequence,
)
from qiskit.transpiler import PassManager, PassManagerConfig, Target
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from .state_preparation import StatePreparationPlugin


@lru_cache(maxsize=1)
def _memoized_basic_approximations(
    basis_gates: FrozenSet[str], depth: int
) -> List[GateSequence]:
    """Generating approximations for Solovay-Kitaev is costly: this function
    caches those approximations"""
    return generate_basic_approximations(basis_gates=basis_gates, depth=depth)


class SKSynthesisPlugin(PassManagerStagePlugin):
    """This plugin configures the Solavay-Kitaev synthesis for the special
    case of logical qubits made out of physical cat qubits.

    Here's what it does:
    * Compute the basis gates to be used for unitary synthesis. This is
      actually different from the basis gates supported by the backend: not
      all backend basis gates can be used as basis gates for the SK synthesis.
      Unfortunately, Qiskit does not allow setting a different basis gate
      set for the synthesis step, so we hack our way around it.
    * Compute approximations for the SK synthesis using only the 1-qubit gates
      of the SK basis gate set
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

        # Compute approximations for the Solovay-Kitaev synthesis
        approximations = _memoized_basic_approximations(
            basis_gates=frozenset(discrete_1q_basis_gates),
            depth=(
                (
                    pass_manager_config.unitary_synthesis_plugin_config or {}
                ).get('depth', 5)
            ),
        )

        # List the gates to synthesize with Solovay-Kitaev. SK only
        # approximates single-qubit rotations, so we restrict the set to
        # 1-qubit gates (minus the discrete basis). Multi-qubit gates are
        # decomposed into the discrete basis plus 1-qubit rotations by
        # ``HighLevelSynthesis``/``BasisTranslator``; those leftover rotations
        # are then discretized by the SK pass that runs after the translator.
        #
        # (In Qiskit 2.x the default unitary-synthesis fallback can no longer
        # decompose an arbitrary multi-qubit unitary against the purely
        # discrete logical target, so routing multi-qubit gates through the
        # equivalence-library passes instead of unitary synthesis is required.)
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

        # Use above SK approximations by default
        pass_manager_config.unitary_synthesis_plugin_config = {
            'basic_approximations': approximations,
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
            # pylint: disable=protected-access
            for task in pm._tasks:
                for subtask in task:
                    if (
                        isinstance(subtask, UnitarySynthesis)
                        and subtask.method == 'sk'
                    ):
                        # There is no option to manually set the gates to
                        # synthesize in UnitarySynthesis, and the default
                        # _synth_gates is just 'unitary'!
                        # pylint: disable=protected-access
                        subtask._synth_gates = synth_gates
                        # Can't pass basis gates in
                        # unitary_synthesis_plugin_config because overridden
                        # by global basis_gates.
                        # These basis gates are used by the default synthesis
                        # method (for multi-qubit unitaries). The
                        # Solovay-Kitaev plugin instead relies on the
                        # precomputed ``basic_approximations`` passed through
                        # the plugin config; ``CustomUnitarySynthesis`` makes
                        # sure the two are not handed to the SK plugin at the
                        # same time.
                        # pylint: disable=protected-access
                        subtask._basis_gates = discrete_basis_gates

        return pm
