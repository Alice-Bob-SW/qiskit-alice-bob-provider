import warnings
from decimal import ROUND_HALF_EVEN, Decimal

from qiskit.circuit import Delay, ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import (
    AnalysisPass,
    PassManager,
    PassManagerConfig,
    TransformationPass,
)
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePlugin

from ..qir import FLOAT_NOISE_NS, NANOSECONDS_PER_UNIT


class DelayInNanosecondsPass(TransformationPass):
    """Converts each delay to a whole number of nanoseconds.

    The conversion uses decimal arithmetic. It rounds to the nearest whole
    number, a half to the even number, and warns at each rounding beyond
    the noise of a float product. A delay
    with an unbound duration becomes a formula in nanoseconds, with no
    rounding.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.op_nodes(Delay):
            duration, unit = node.op.duration, node.op.unit
            if unit not in NANOSECONDS_PER_UNIT:
                raise TranspilerError(
                    f'The delay of {duration} {unit} has no duration in '
                    'nanoseconds. Give each delay in a unit of time: s, ms, '
                    'us, ns or ps.'
                )
            factor = NANOSECONDS_PER_UNIT[unit]
            if isinstance(duration, ParameterExpression):
                nanoseconds = duration * factor.numerator / factor.denominator
            else:
                exact = (
                    Decimal(str(duration))
                    * factor.numerator
                    / factor.denominator
                )
                nanoseconds = int(
                    exact.to_integral_value(rounding=ROUND_HALF_EVEN)
                )
                if abs(exact - nanoseconds) > FLOAT_NOISE_NS:
                    warnings.warn(
                        f'Rounded the delay of {exact.normalize():f} ns to '
                        f'{nanoseconds} ns. A delay is a whole number of '
                        'nanoseconds.'
                    )
            dag.substitute_node(node, Delay(nanoseconds, unit='ns'))
        return dag


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
    """The conversion of each delay to whole nanoseconds, then a warning with
    the count of Q gates of the transpiled circuit.

    The scheduling stage runs last, so the count holds after the
    optimizations. This stage replaces the default one of Qiskit, which
    turns each delay into float seconds: `delay(0.0125, unit='us')` then
    becomes 12.500000000000001 ns and rounds to 13 instead of 12. The
    remote targets have no timing constraint, so the default stage has
    nothing else to do.
    """

    def pass_manager(
        self,
        pass_manager_config: PassManagerConfig,
        optimization_level=None,
    ) -> PassManager:
        return PassManager([DelayInNanosecondsPass(), QGateWarningPass()])
