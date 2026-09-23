import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit_alice_bob_provider.custom_instructions import MeasureX

OPTIMIZATION_LEVELS = [0, 1, 2, 3]


def _physical_cats_backend():
    return AliceBobLocalProvider().get_backend('EMU:6Q:PHYSICAL_CATS')


def test_measure_x_method_appends_measure_x() -> None:
    circ = QuantumCircuit(2, 2)
    circ.measure_x(1, 0)
    (inst,) = circ.data
    assert isinstance(inst.operation, MeasureX)
    assert circ.find_bit(inst.qubits[0]).index == 1
    assert circ.find_bit(inst.clbits[0]).index == 0


def test_measure_x_lowers_to_basis_without_measure_x() -> None:
    circ = QuantumCircuit(1, 1)
    circ.append(MeasureX(), [0], [0])
    transpiled = transpile(
        circ,
        basis_gates=['h', 'measure', 'rz', 'sx', 'x', 'cx'],
        optimization_level=0,
    )
    assert [inst.operation.name for inst in transpiled.data] == [
        'h',
        'measure',
        'h',
    ]


@pytest.mark.parametrize('state,outcome', [('+', '0'), ('-', '1')])
def test_measure_x_measures_in_x_basis_on_aer(
    state: str, outcome: str
) -> None:
    circ = QuantumCircuit(1, 1)
    circ.initialize(state, 0)
    circ.measure_x(0, 0)
    backend = AerSimulator()
    result = backend.run(transpile(circ, backend), shots=100).result()
    assert result.get_counts() == {outcome: 100}


@pytest.mark.parametrize('optimization_level', OPTIMIZATION_LEVELS)
def test_measure_x_is_kept_when_target_supports_it(
    optimization_level: int,
) -> None:
    circ = QuantumCircuit(1, 1)
    circ.initialize('+', 0)
    circ.measure_x(0, 0)
    ops = transpile(
        circ, _physical_cats_backend(), optimization_level=optimization_level
    ).count_ops()
    assert ops['measure_x'] == 1
    assert 'h' not in ops
    assert 'measure' not in ops


@pytest.mark.parametrize('optimization_level', OPTIMIZATION_LEVELS)
def test_reset_becomes_initialize_zero(optimization_level: int) -> None:
    circ = QuantumCircuit(1, 1)
    circ.x(0)
    circ.reset(0)
    circ.measure(0, 0)
    transpiled = transpile(
        circ, _physical_cats_backend(), optimization_level=optimization_level
    )
    ops = [
        inst.operation
        for inst in transpiled.data
        if inst.operation.name != 'delay'
    ]
    names = [op.name for op in ops]
    assert 'reset' not in names
    after_x = ops[names.index('x') + 1]
    assert after_x.name == 'initialize'
    assert list(after_x.params) == ['0']
