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

import numpy as np
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.transpiler import (
    PassManager,
    PassManagerConfig,
    TransformationPass,
    TranspilerError,
)
from qiskit.transpiler.preset_passmanagers.common import (
    generate_translation_passmanager,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from ..ensure_preparation_pass import EnsurePreparationPass


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
