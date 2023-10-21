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
from typing import FrozenSet, List, Set

import numpy as np
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping,
)
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.synthesis.discrete_basis.gate_sequence import GateSequence
from qiskit.synthesis.discrete_basis.solovay_kitaev import (
    generate_basic_approximations,
)
from qiskit.transpiler import (
    PassManager,
    PassManagerConfig,
    TransformationPass,
    TranspilerError,
)
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler.preset_passmanagers.common import (
    generate_translation_passmanager,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from ..ensure_preparation_pass import EnsurePreparationPass
from ..processor.logical_cat import LogicalCatProcessor
from .proc_to_qiskit import processor_to_qiskit_instruction


class IntToLabelInitializePass(TransformationPass):
    """
    A transpilation pass that transforms intializations using integers into
    intializations using labels.

    Example:
    ```
    circ = QuantumCircuit(2)
    circ.initialize(2)
    transpiled = pm.run(circ)
    print(circ)
    #      ┌────────────────┐
    # q_0: ┤0               ├
    #     │  Initialize(2) │
    # q_1: ┤1               ├
    #     └────────────────┘
    print(transpiled)
    #     ┌──────────────────┐
    # q_0: ┤0                 ├
    #     │  Initialize(1,0) │
    # q_1: ┤1                 ├
    #     └──────────────────┘
    ```
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if node.name != 'initialize':
                continue
            assert isinstance(node.op, Initialize)
            params = node.op.params
            if len(params) > 1:
                if isinstance(params[0], str):
                    continue
                raise TranspilerError(
                    'State vectors are not supported as input of initialize'
                    f' (params are {params}). Please use string labels like'
                    ' Initialize("+")'
                )
            assert len(params) == 1
            if not isinstance(params[0], (int, float, complex)):
                continue
            integer_state = int(np.real(params[0]))
            label_state = f'{integer_state:0{int(node.op.num_qubits)}b}'
            dag.substitute_node(
                node,
                Initialize(label_state),
            )
        return dag


class BreakDownInitializePass(TransformationPass):
    """
    A transpilation pass that transforms label, multi-qubit initializations
    into label, single-qubit initializations.

    Example:
    ```
    circ = QuantumCircuit(2)
    circ.initialize('10')
    transpiled = pm.run(circ)
    print(circ)
    #     ┌──────────────────┐
    # q_0: ┤0                 ├
    #     │  Initialize(1,0) │
    # q_1: ┤1                 ├
    #     └──────────────────┘
    print(transpiled)
    #     ┌───────────────┐
    # q_0: ┤ Initialize(0) ├
    #     ├───────────────┤
    # q_1: ┤ Initialize(1) ├
    #     └───────────────┘
    ```
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if node.name != 'initialize':
                continue
            assert isinstance(node.op, Initialize)
            params = node.op.params
            assert len(params) > 0
            if not isinstance(params[0], str):
                raise TranspilerError(
                    'At this stage in the transpilation process, only label'
                    f' initializations should exist (params are {params})'
                )
            new_dag = DAGCircuit()
            new_dag.add_qubits(node.qargs)
            new_dag.add_clbits(node.cargs)
            for index, state in enumerate(params[::-1]):
                new_dag.apply_operation_back(
                    Initialize(state), qargs=(node.qargs[index],)
                )
            dag.substitute_node_with_dag(node, new_dag)
        return dag


class LocalStatePreparationPlugin(PassManagerStagePlugin):
    """A pass manager meant to be used as a translation plugin that ensures
    all qubits are initialized with a known state among 0, 1, +, -.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        custom_pm = PassManager()
        custom_pm.append(EnsurePreparationPass(lambda: Initialize('0')))
        custom_pm.append(IntToLabelInitializePass())
        custom_pm.append(BreakDownInitializePass())

        default_pm = generate_translation_passmanager(
            target=pass_manager_config.target,
            basis_gates=pass_manager_config.basis_gates,
            method='translator',
            approximation_degree=pass_manager_config.approximation_degree,
            coupling_map=pass_manager_config.coupling_map,
            backend_props=pass_manager_config.backend_properties,
            unitary_synthesis_method=(
                pass_manager_config.unitary_synthesis_method
            ),
            unitary_synthesis_plugin_config=(
                pass_manager_config.unitary_synthesis_plugin_config
            ),
            hls_config=pass_manager_config.hls_config,
        )
        for passes in default_pm.passes():
            for p in passes.values():
                custom_pm.append(p)

        return custom_pm


@lru_cache(maxsize=1)
def _memoized_basic_approximations(
    basis_gates: FrozenSet[str], depth: int
) -> List[GateSequence]:
    """Generating approximations for Solovay-Kitaev is costly: this function
    caches those approximations"""
    return generate_basic_approximations(basis_gates=basis_gates, depth=depth)


class LocalLogicalCatPlugin(PassManagerStagePlugin):
    """This plugin configures the Solavay-Kitaev synthesis for the special
    case of logical qubits made out of physical cat qubits.

    Here's what it does:
    * Compute the basis gates to be used for unitary synthesis. This is
      actually different from the basis gates supported by the backend: not
      all backend basis gates can be used as basis gates for the SK synthesis.
      Unfortunately, Qiskit does not allow setting a different basis gate
      set for te synthesis step, so we hack our way around it.
    * Compute approximations for the SK synthesis using only the 1-qubit gates
      of the SK basis gate set
    * Mix the SK synthesis with the transpilation passes from
      LocalStatePreparationPlugin

    This plugin wouldn't exist if Qiskit's transpile exposed more options to
    configure the synthesis method (and didn't override the available options
    in strange ways)."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        # Compute the discrete basis gates for the Solovay-Kitaev synthesis
        proc = LogicalCatProcessor()
        discrete_basis_gates: Set[str] = set()
        discrete_1q_basis_gates: Set[str] = set()
        for instr in proc.all_instructions():
            qiskit_instr = processor_to_qiskit_instruction(instr)
            if len(qiskit_instr.params) != 0 or qiskit_instr.name in {
                'measure',
                'measure_x',
                'delay',
            }:
                continue
            if qiskit_instr.num_qubits == 1:
                discrete_1q_basis_gates.add(qiskit_instr.name)
            discrete_basis_gates.add(qiskit_instr.name)

        # Compute approximations for the Solovay-Kitaev synthesis
        approximations = _memoized_basic_approximations(
            basis_gates=frozenset(discrete_1q_basis_gates),
            depth=(
                (
                    pass_manager_config.unitary_synthesis_plugin_config or {}
                ).get('depth', 5)
            ),
        )

        # List gates to synthesize (all gates except basis gates)
        synth_gates: Set[str] = set()
        for name, instr in get_standard_gate_name_mapping().items():
            if name in {'measure', 'measure_x', 'reset', 'delay'}:
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
        pm = LocalStatePreparationPlugin().pass_manager(
            pass_manager_config=pass_manager_config,
            optimization_level=optimization_level,
        )

        # Update the UnitarySynthesis pass (which controls the
        # SolovayKitaevSynthesis plugin) with the gates to synthesize and basis
        # gates computed above.
        if pass_manager_config.unitary_synthesis_method == 'sk':
            for passess in pm.passes():
                for passes in passess.values():
                    for p in passes:
                        if (
                            isinstance(p, UnitarySynthesis)
                            and p.method == 'sk'
                        ):
                            # There is no option to manually set the gates to
                            # synthesize in UnitarySynthesis, and the default
                            # _synth_gates is just 'unitary'!
                            # pylint: disable=protected-access
                            p._synth_gates = synth_gates
                            # Can't pass basis gates in
                            # unitary_synthesis_plugin_config because overriden
                            # by global basis_gates.
                            # This seems to be a Qiskit bug (why give the
                            # possibility to specify basis gates in the plugin
                            # config if that was not the intent?)
                            # pylint: disable=protected-access
                            p._basis_gates = discrete_basis_gates

        return pm
