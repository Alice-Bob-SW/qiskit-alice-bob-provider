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
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.passes import ASAPScheduleAnalysis, PadDelay
from qiskit.transpiler.preset_passmanagers.common import generate_scheduling
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


class ProcessorASAPScheduleAnalysis(ASAPScheduleAnalysis):
    """ASAP scheduling analysis that resolves instruction durations from the
    full operation (with its bound parameters) instead of the gate name only.

    Since Qiskit 2.0, ``BaseScheduler._get_node_duration`` queries durations
    with just the instruction *name*
    (``self.durations.get(node.name, indices)``). Alice & Bob's
    :class:`ProcessorInstructionDurations` computes parameter-dependent
    durations and needs the operation's parameters, both to disambiguate
    instructions that share a Qiskit name (e.g. ``Initialize('0')`` vs
    ``Initialize('+')``) and to evaluate continuously parameter-dependent
    durations (e.g. ``rz(theta)``). We therefore override the lookup to pass
    ``node.op``.
    """

    def _get_node_duration(self, node: DAGOpNode, dag: DAGCircuit) -> int:
        if node.name == 'delay':
            # `TimeUnitConversion` already handled the unit conversion.
            return node.op.duration

        indices = [dag.find_bit(qarg).index for qarg in node.qargs]
        unit = 's' if self.durations.dt is None else 'dt'
        duration = self.durations.get(node.op, indices, unit=unit)

        if isinstance(duration, ParameterExpression):
            duration = duration.numeric()
        return duration


class ProcessorPadDelay(PadDelay):
    """Delay-padding pass that resolves durations from the full operation.

    Like :class:`ProcessorASAPScheduleAnalysis`, the upstream
    ``BasePadding.get_duration`` looks durations up by gate name (and prefers
    ``Target`` instruction properties, which do not carry the Alice & Bob
    durations). We override it to query the
    :class:`ProcessorInstructionDurations` with ``node.op`` instead.
    """

    def get_duration(self, node: DAGOpNode, dag: DAGCircuit) -> int:
        if node.name == 'delay':
            return node.op.duration
        if node.name == 'barrier':
            return 0
        indices = [dag.find_bit(qarg).index for qarg in node.qargs]
        duration = self.durations.get(node.op, indices)
        if isinstance(duration, ParameterExpression):
            duration = duration.numeric()
        return duration


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

        # `generate_scheduling` hardcodes the upstream `ASAPScheduleAnalysis`
        # and `PadDelay`, neither of which can read Alice & Bob's
        # parameter-dependent durations. Replace them with our processor-aware
        # variants (and teach the scheduler to handle control-flow operations).
        # pylint: disable=protected-access
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
                # Unlike `BaseScheduler`, `BasePadding` keeps the `durations`
                # argument as-is instead of preferring `target.durations()`.
                # We therefore hand it the processor durations explicitly (the
                # dynamic ones live on the target, not in
                # `pass_manager_config.instruction_durations`).
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
