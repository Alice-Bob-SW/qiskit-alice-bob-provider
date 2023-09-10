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

from typing import Dict, Iterator, List, Tuple

import numpy as np

from qiskit_alice_bob_provider.processor.description import (
    AppliedInstruction,
    InstructionProperties,
)

from .description import ProcessorDescription
from .serialization.interpolate import InterpolatedProcessor
from .serialization.model import SerializedInstruction, SerializedProcessor


class InterpolatedCatProcessor(ProcessorDescription):
    """A type of processor description built from a serialized representation
    of the processor behavior.

    Compared to the more generic class ``InterpolatedProcessor``,
    ``InterpolatedCatProcessor`` is tailored to cat qubit based processors.
    In particular, it automatically injects alpha/nbar before the interpolation
    step.
    """

    def __init__(
        self,
        serialized_processor: SerializedProcessor,
        clock_cycle: float = 1e-9,
        alpha: float = 2,
    ) -> None:
        self._interpolated_processor = InterpolatedProcessor(
            serialized_processor, clock_cycle=clock_cycle
        )
        self.clock_cycle = clock_cycle
        self._alpha = alpha
        self._instructions: Dict[
            Tuple[str, Tuple[int, ...]], SerializedInstruction
        ] = {}
        for instr in serialized_processor.instructions:
            for qubits in instr.qubits:
                self._instructions[(instr.name, tuple(qubits))] = instr

    def all_instructions(self) -> Iterator[InstructionProperties]:
        for instr in self._interpolated_processor.all_instructions():
            yield InstructionProperties(
                name=instr.name,
                qubits=instr.qubits,
                params=[p for p in instr.params if p != 'nbar'],
                readout_errors=instr.readout_errors,
            )

    def apply_instruction(
        self, name: str, qubits: Tuple[int, ...], params: List[float]
    ) -> AppliedInstruction:
        instr = self._instructions[(name, qubits)]
        idx = 0
        new_params = []
        for param_name in instr.free_params:
            if param_name == 'nbar':
                new_params.append(np.abs(self._alpha) ** 2)
            else:
                new_params.append(params[idx])
                idx += 1
        return self._interpolated_processor.apply_instruction(
            name, qubits, new_params
        )
