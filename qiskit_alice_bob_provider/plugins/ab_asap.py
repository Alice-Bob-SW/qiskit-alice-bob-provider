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

from qiskit.circuit import ControlFlowOp
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import (
    InstructionDurations,
    PassManager,
    PassManagerConfig,
    TranspilerError,
)
from qiskit.transpiler.passes import ASAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers.common import generate_scheduling
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


def _node_duration(
    durations: InstructionDurations,
    node: DAGOpNode,
    dag: DAGCircuit,
    unit: str,
) -> float:
    indices = [dag.find_bit(qarg).index for qarg in node.qargs]
    if node.name == 'delay':
        # TimeUnitConversion converts the delay unit before scheduling.
        duration = node.op.duration
    else:
        duration = durations.get(node.op, indices, unit=unit)

    if isinstance(duration, ParameterExpression):
        try:
            duration = duration.numeric()
        except TypeError as exc:
            names = ', '.join(sorted(p.name for p in duration.parameters))
            raise TranspilerError(
                f'The duration of {node.op.name} on qubits {indices} depends '
                f'on the unbound parameters {names}. Bind the parameters '
                'with assign_parameters before transpile.'
            ) from exc
    return duration


class ProcessorASAPScheduleAnalysis(ASAPScheduleAnalysis):
    """ASAP scheduling analysis that computes each duration from the full
    operation.

    The upstream pass finds a duration from the instruction name only. A
    processor duration can also depend on the operation parameters, for
    example the angle of an rz gate or the state of an Initialize.
    """

    def _get_node_duration(self, node: DAGOpNode, dag: DAGCircuit) -> float:
        unit = 's' if self.durations.dt is None else 'dt'
        return _node_duration(self.durations, node, dag, unit)


class ProcessorPadDelay(PadDelay):
    """Delay padding pass that computes each duration from the full
    operation.

    The upstream pass finds a duration from the instruction name only.
    """

    def get_duration(self, node: DAGOpNode, dag: DAGCircuit) -> float:
        if node.name == 'barrier':
            return 0
        return _node_duration(self.durations, node, dag, 'dt')


class AliceBobASAPSchedulingPlugin(PassManagerStagePlugin):
    """A pass manager to compute scheduling of a circuit to run on
    Alice & Bob's local backends with the ASAP strategy."""

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        pm = generate_scheduling(
            instruction_durations=pass_manager_config.instruction_durations,
            scheduling_method='asap',
            timing_constraints=pass_manager_config.timing_constraints,
            target=pass_manager_config.target,
        )

        # generate_scheduling always uses the upstream ASAPScheduleAnalysis
        # and PadDelay. These passes cannot compute a duration that depends on
        # the operation parameters, so replace them.
        for index, task in enumerate(pm._tasks):
            if any(
                isinstance(subtask, ASAPScheduleAnalysis) for subtask in task
            ):
                scheduler = ProcessorASAPScheduleAnalysis(
                    pass_manager_config.instruction_durations,
                    target=pass_manager_config.target,
                )
                # By default, the scheduling analysis pass only supports
                # conditionals for Gate & Delay instructions. We add support
                # for ControlFlowOp (which itself covers If-Else, For-Loop,
                # While-Loop and Switch-Case).
                scheduler.CONDITIONAL_SUPPORTED = (
                    *scheduler.CONDITIONAL_SUPPORTED,
                    ControlFlowOp,
                )
                pm.replace(index, scheduler)
            elif any(isinstance(subtask, PadDelay) for subtask in task):
                # PadDelay does not read the durations of the target, which
                # hold the processor durations. Pass them explicitly.
                target = pass_manager_config.target
                durations = (
                    target.durations()
                    if target is not None
                    else pass_manager_config.instruction_durations
                )
                pm.replace(
                    index,
                    ProcessorPadDelay(target=target, durations=durations),
                )

        return pm
