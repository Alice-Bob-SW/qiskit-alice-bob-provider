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

"""Custom optimization stage plugin that avoids the re-translation oscillation
triggered by Qiskit 1.4.3+ changes to SolovayKitaev synthesis.

The default Qiskit optimization stage (level >= 1) contains a DoWhileController
loop that re-runs the full translation plugin (sk_synthesis) via a
ConditionalController when GatesInBasis detects out-of-basis gates. The SK
synthesis changes in Qiskit 1.4.3 cause this re-translation to produce
different results each iteration, leading to oscillation and the error
'Maximum iteration reached. max_iteration=1000'.

This plugin keeps useful passes (InverseCancellation) but removes the
problematic GatesInBasis + ConditionalController(translation) re-translation
step and Optimize1qGatesDecomposition (which can produce out-of-basis gates
for discrete targets per Qiskit bug #14777).
"""

from typing import Optional

from qiskit.circuit.library import (
    CCXGate,
    CXGate,
    CZGate,
    ECRGate,
    HGate,
    SGate,
    SdgGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.passmanager import DoWhileController
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.passes import (
    Depth,
    FixedPoint,
    InverseCancellation,
    Size,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


class AliceBobOptimizationPlugin(PassManagerStagePlugin):
    """Optimization stage plugin for Alice & Bob backends using SK synthesis.

    This plugin replaces the default Qiskit optimization stage to avoid the
    re-translation oscillation that occurs with Qiskit 1.4.3+ when using
    the sk_synthesis translation plugin.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level: Optional[int] = None,
    ) -> Optional[PassManager]:
        if optimization_level == 0:
            return None

        inverse_cancellation = InverseCancellation(
            [
                CXGate(),
                CZGate(),
                XGate(),
                YGate(),
                ZGate(),
                HGate(),
                SwapGate(),
                CCXGate(),
                ECRGate(),
                (TGate(), TdgGate()),
                (SGate(), SdgGate()),
                (SXGate(), SXdgGate()),
            ]
        )

        pm = PassManager()
        pm.append(
            DoWhileController(
                [
                    Depth(recurse=True),
                    Size(recurse=True),
                    inverse_cancellation,
                    Depth(recurse=True),
                    Size(recurse=True),
                    FixedPoint('depth'),
                    FixedPoint('size'),
                ],
                do_while=_not_converged,
            )
        )
        return pm


def _not_converged(property_set):
    return not (
        property_set['depth_fixed_point'] and property_set['size_fixed_point']
    )
