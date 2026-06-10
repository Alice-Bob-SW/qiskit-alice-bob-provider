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
from copy import deepcopy
from typing import Callable

import numpy as np
from qiskit.circuit import Instruction, Qubit, Reset
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import Initialize, RZGate
from qiskit.dagcircuit import DAGCircuit, DAGInNode, DAGOpNode
from qiskit.transpiler import (
    PassManager,
    PassManagerConfig,
    TransformationPass,
    TranspilerError,
)
from qiskit.transpiler.passes import (
    BasisTranslator,
    HighLevelSynthesis,
    TrivialLayout,
    UnitarySynthesis,
    UnrollCustomDefinitions,
)
from qiskit.transpiler.preset_passmanagers.common import (
    generate_embed_passmanager,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


def _reset_prep() -> Instruction:
    return Reset()


class CustomUnitarySynthesis(UnitarySynthesis):
    """``UnitarySynthesis`` that avoids handing a plugin both ``basis_gates``
    and precomputed ``basic_approximations``.

    Two things changed in Qiskit 2.x that interact badly here:

    * ``UnitarySynthesis._run_main_loop`` now synthesizes every node whose name
      is in ``self._synth_gates`` (it used to be hardcoded to ``"unitary"``),
      which is exactly the behaviour the Alice & Bob fork used to patch in.
    * ``SolovayKitaevDecomposition`` now raises if it is given both
      ``basis_gates`` and ``basic_approximations``.

    The pass still needs ``basis_gates`` for the *default* synthesis method
    (used for multi-qubit unitaries), so we keep them on the pass and only
    drop them from the kwargs of a plugin that was already supplied with
    ``basic_approximations`` (i.e. the Solovay-Kitaev plugin configured by
    :class:`SKSynthesisPlugin`).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _run_plugin_synthesis(
        self, method, kwargs, unitary, qubit_indices, qargs
    ):
        config = kwargs.get('config') or {}
        if (
            config.get('basic_approximations') is not None
            and 'basis_gates' in kwargs
        ):
            kwargs = {k: v for k, v in kwargs.items() if k != 'basis_gates'}
        return super()._run_plugin_synthesis(
            method, kwargs, unitary, qubit_indices, qargs
        )


def _get_unitary_synthesis(config: PassManagerConfig):
    return CustomUnitarySynthesis(
        config.basis_gates,
        approximation_degree=config.approximation_degree,
        coupling_map=config.coupling_map,
        plugin_config=config.unitary_synthesis_plugin_config,
        method=config.unitary_synthesis_method,
        target=config.target,
    )


def _get_high_level_synthesis(config: PassManagerConfig):
    return HighLevelSynthesis(
        hls_config=config.hls_config,
        coupling_map=config.coupling_map,
        target=config.target,
        use_qubit_indices=True,
        equivalence_library=SessionEquivalenceLibrary,
        basis_gates=config.basis_gates,
    )


class EnsurePreparationPass(TransformationPass):
    """
    A transpilation pass that ensures there is a state preparation at the
    beginning of the circuit.

    If there isn't, a Reset is added (i.e. prepare the zero state in the
    Z basis).

    The reason for this pass is that Qiskit runs the `RemoveResetInZeroState`
    pass as part of the default transpilation passes, in most cases (precisely
    for optimization_level >=1, the default being 1).

    The community seems to agree that this behavior is a bad decision and
    should be removed. https://github.com/Qiskit/qiskit-terra/issues/6943
    The rationale for this behavior was that IBM backends always reset the
    qubits to zero between shots. This behavior cannot be assumed to be true
    for all backends, including Alice & Bob's.
    Once this issue is resolved, this transpilation pass should be removed from
    the Alice & Bob provider.

    This transpilation pass cannot be added to the transpilation stage plugin
    of the Alice & Bob provider, because the `RemoveResetInZeroState` runs
    in the optimization stage, which happens after the transpilation stage.

    For this reason, this pass is run manually right before the compilation
    from Qiskit to QIR.
    """

    def __init__(
        self, prep_instruction: Callable[[], Instruction] = _reset_prep
    ):
        super().__init__()
        self._prep = prep_instruction

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        preparations = {'reset', 'initialize', 'state_preparation'}
        for node in dag.topological_nodes():
            if not isinstance(node, DAGInNode):
                continue
            if not isinstance(node.wire, Qubit):
                continue
            for successor in dag.successors(node):
                if not isinstance(successor, DAGOpNode):
                    continue
                if successor.name in preparations:
                    continue
                new_dag = DAGCircuit()
                new_dag.add_qubits(successor.qargs)
                new_dag.add_clbits(successor.cargs)
                new_dag.apply_operation_back(self._prep(), qargs=(node.wire,))
                new_dag.apply_operation_back(
                    successor.op,
                    qargs=successor.qargs,
                    cargs=successor.cargs,
                )
                dag.substitute_node_with_dag(successor, new_dag)
        return dag


class IntToLabelInitializePass(TransformationPass):
    """
    A transpilation pass that transforms initializations using integers into
    initializations using labels.

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


class StatePreparationPlugin(PassManagerStagePlugin):
    """A translation plugin built on top of the base translator plugin with
    the extra operations:
    - Apply a Layout embedding from the target's coupling map.
    - Ensure all qubits are initialized with a known state among 0, 1, +, -.
    - Format and break down the initialize gates.
    - Ensure we unroll custom gate definitions before applying synthesis.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        custom_pm = PassManager()
        custom_pm.append(TrivialLayout(pass_manager_config.target))
        custom_pm.append(EnsurePreparationPass(lambda: Initialize('0')))
        custom_pm.append(IntToLabelInitializePass())
        custom_pm.append(BreakDownInitializePass())

        # Can probably be removed (replaced by HighLevelSynthesis?)
        custom_pm.append(
            UnrollCustomDefinitions(
                equivalence_library=SessionEquivalenceLibrary,
                basis_gates=pass_manager_config.basis_gates,
                target=pass_manager_config.target,
            )
        )

        # Replace passes from qiskit generate_translation_passmanager() with
        # passes that work for us.
        # By default, qiskit returns
        #   [UnitarySynthesis, HighLevelSynthesis, BasisTranslator]
        # In our case, we need
        # 1. to replace UnitarySynthesis with our CustomUnitarySynthesis class
        # 2. to adapt the passes to handle the "Clifford + T gate basis" case

        if 'rz' in pass_manager_config.target:
            # In this case, no need to modify the BasisTranslator pass and to
            # force the SK synthesis after.
            need_synthesis = False
            basis_translator_target = pass_manager_config.target
        else:
            # In this case, our basis target consists of Clifford + T gate,
            # i.e. {cx, h, s, t}. It does not contain unitary rotations
            # (e.g. rz).
            #
            # This is a universal set of gates, but it requires synthesis
            # (e.g. with SK algorithm), for transpilation to succeed. This
            # synthesis (that transforms rotations into discrete gates of our
            # basis target) is done during the CustomUnitarySynthesis pass.
            #
            # Unfortunately, qiskit BasisTranslator pass does not handle
            # synthesis. It tries to match existing gates in the circuit to
            # gates in the target basis, but only for exact equivalence (doing
            # a graph search).
            #
            # Therefore, as a workaround, when the target basis does not
            # support unitary rotations natively, we
            #   - add the 'rz' to the BasisTranslator target basis set, to
            #     trick the algorithm into thinking that rotations are
            #     supported by the target.
            #   - add a 2nd CustomUnitarySynthesis pass after that, to get
            #     rid of any 'rz' pass generated in the pass above.
            #
            # Example :
            # > Input circuit ('cs')
            # q_0: ──■──
            #      ┌─┴─┐
            # q_1: ┤ S ├
            #      └───┘
            #
            # > output of BasisTranslator, with 'rz' in target :
            #      ┌─────────┐
            # q_0: ┤ Rz(π/4) ├──■────────────────■─────────────
            #      └─────────┘┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐
            # q_1: ───────────┤ X ├┤ Rz(-π/4) ├┤ X ├┤ Rz(π/4) ├
            #                 └───┘└──────────┘└───┘└─────────┘
            #
            # > output of BasisTranslator, without 'rz' in target :
            # TranspilerError: "Unable to translate the operations..."
            #
            # > output of the 2nd CustomUnitarySynthesis :
            #      ┌───┐
            # q_0: ┤ T ├──■───────────■───────
            #      └───┘┌─┴─┐┌─────┐┌─┴─┐┌───┐
            # q_1: ─────┤ X ├┤ Tdg ├┤ X ├┤ T ├
            #           └───┘└─────┘└───┘└───┘
            need_synthesis = True

            # Ideally, we would use the `target_basis` argument of
            # BasisTranslator to specify we want to add the "rz" gate.
            # Unfortunately, in the .run method, this argument is ignored in
            # favor of the target.target_basis when the target is defined.
            # Therefore, we create a copy of the current target and modify
            # its set of supported instructions just for this pass.
            basis_translator_target = deepcopy(pass_manager_config.target)
            basis_translator_target.add_instruction(RZGate(0))

        custom_pm.append(_get_unitary_synthesis(pass_manager_config))
        custom_pm.append(_get_high_level_synthesis(pass_manager_config))
        custom_pm.append(
            BasisTranslator(
                SessionEquivalenceLibrary,
                pass_manager_config.basis_gates,
                basis_translator_target,
            ),
        )
        if need_synthesis:
            custom_pm.append(_get_unitary_synthesis(pass_manager_config))

        custom_pm += generate_embed_passmanager(
            pass_manager_config.coupling_map
        )
        return custom_pm
