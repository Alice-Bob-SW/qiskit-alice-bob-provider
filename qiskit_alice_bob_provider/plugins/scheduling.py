import warnings

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass, PassManager, PassManagerConfig
from qiskit.transpiler.preset_passmanagers.builtin_plugins import (
    DefaultSchedulingPassManager,
)
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin


def warn_q_gates(count: int, subject: str) -> None:
    """Warn that `subject` holds `count` Q gates (`sx`), if it holds any.

    A Q gate turns the phase flips of a cat qubit into bit flips.
    """
    if count == 0:
        return
    gates = 'Q gate' if count == 1 else 'Q gates'
    warnings.warn(
        f'{subject} holds {count} {gates} (sx). A Q gate breaks the noise '
        'bias of a cat qubit.'
    )


class QGateWarningPass(AnalysisPass):
    """Warns with the count of Q gates of the circuit."""

    def run(self, dag: DAGCircuit) -> None:
        count = dag.count_ops(recurse=True).get('sx', 0)
        warn_q_gates(count, 'The transpiled circuit')


class AliceBobSchedulingPlugin(PassManagerStagePlugin):
    """The default scheduling stage of Qiskit, then a warning with the count
    of Q gates of the transpiled circuit.

    The scheduling stage runs last, so the count holds after the
    optimizations.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        scheduling = DefaultSchedulingPassManager().pass_manager(
            pass_manager_config, optimization_level
        )
        scheduling.append(QGateWarningPass())
        return scheduling
