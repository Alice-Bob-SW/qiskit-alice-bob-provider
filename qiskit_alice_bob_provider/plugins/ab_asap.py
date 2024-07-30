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

from typing import cast

from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.preset_passmanagers.common import generate_scheduling
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from qiskit_alice_bob_provider.local.instruction_durations import (
    ProcessorInstructionDurations,
)


class CustomTimeUnitConversion(TimeUnitConversion):
    """
    This is a replacement of the base TimeUnitConversion pass to handle the
    ProcessorInstructionDurations object created for Alice & Bob backends.

    The pass overrides the _update_inst_durations method to apply the base
    durations from Alice & Bob's ProcessorInstructionDuration instead of
    the generic InstructionDurations class from Qiskit.
    """

    def _update_inst_durations(self, dag) -> InstructionDurations:
        """Update instruction durations with circuit information.
        If the dag contains gate calibrations and no instruction durations were
        provided through the target or as a standalone input, the circuit
        calibration durations will be used. The priority order for
        instruction durations is: target > standalone > circuit.
        """
        if not isinstance(self.inst_durations, ProcessorInstructionDurations):
            return self.inst_durations

        # pylint: disable=protected-access
        circ_durations = ProcessorInstructionDurations(
            self.inst_durations._proc
        )

        if dag.calibrations:
            cal_durations = []
            for gate, gate_cals in dag.calibrations.items():
                for (qubits, parameters), schedule in gate_cals.items():
                    cal_durations.append(
                        (gate, qubits, parameters, schedule.duration)
                    )
            circ_durations.update(cal_durations, circ_durations.dt)

        if self._durations_provided:
            # We cast dt to float for mypy, to match the signature required by
            # update(), but the real type of dt is indeed Optional[float].
            circ_durations.update(
                self.inst_durations,
                cast(float, getattr(self.inst_durations, 'dt', None)),
            )

        return circ_durations


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
            inst_map=pass_manager_config.inst_map,
            target=pass_manager_config.target,
        )

        # pylint: disable=protected-access
        for task in pm._tasks:
            for i, subtask in enumerate(task):
                # Substitue the default TimeUnitConversion with our
                # custom implementation.
                if isinstance(subtask, TimeUnitConversion):
                    pm.replace(
                        i,
                        CustomTimeUnitConversion(
                            target=pass_manager_config.target
                        ),
                    )

        return pm
