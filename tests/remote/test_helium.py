##############################################################################
# Copyright 2026 Alice & Bob
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

import re
import warnings
from math import pi
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Initialize, SXGate
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import PassManager
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.plugins.state_preparation import (
    LowerXYBasisPass,
)
from qiskit_alice_bob_provider.qir import circuit_to_qir
from qiskit_alice_bob_provider.remote.provider import AliceBobRemoteProvider
from qiskit_alice_bob_provider.remote.qir_to_qiskit import (
    ab_target_to_qiskit_target,
)


def _helium_backend(helium_target: Dict, requests_mock: Mocker):
    requests_mock.register_uri('GET', '/v1/targets/', json=[helium_target])
    return AliceBobRemoteProvider(api_key='foo').get_backend(
        helium_target['name']
    )


def _lower(helium_target: Dict, circuit: QuantumCircuit) -> QuantumCircuit:
    target = ab_target_to_qiskit_target(helium_target)
    return PassManager([LowerXYBasisPass(target)]).run(circuit)


def _evolve_after_preparation(circuit: QuantumCircuit) -> Statevector:
    first, *rest = circuit.data
    assert isinstance(first.operation, Initialize)
    state = Statevector.from_label(first.operation.params[0])
    for instruction in rest:
        state = state.evolve(instruction.operation)
    return state


def test_sx_has_the_matrix_of_qiskit_sx(helium_target: Dict) -> None:
    target = ab_target_to_qiskit_target(helium_target)
    assert Operator(target.operation_from_name('sx')) == Operator(SXGate())


def test_announcement_builds_without_warning(helium_target: Dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ab_target_to_qiskit_target(helium_target)


@pytest.mark.parametrize(
    'pair, is_wired',
    [
        ((0, 1), True),
        ((2, 1), True),
        ((0, 17), True),
        ((1, 0), False),
        ((1, 2), False),
        ((0, 3), False),
    ],
)
def test_cx_follows_the_wiring(
    helium_target: Dict, pair: Tuple[int, int], is_wired: bool
) -> None:
    target = ab_target_to_qiskit_target(helium_target)
    assert target.instruction_supported('cx', pair) is is_wired


def test_cz_follows_the_wiring(helium_target: Dict) -> None:
    cz = {'signature': '__quantum__qis__cz__body:void (%Qubit*, %Qubit*)'}
    target = ab_target_to_qiskit_target(
        {**helium_target, 'instructions': [cz]}
    )
    assert target.instruction_supported('cz', (0, 1))
    assert not target.instruction_supported('cz', (1, 0))


def test_cx_without_connectivity_runs_on_every_pair(
    all_instructions_target: Dict,
) -> None:
    target = ab_target_to_qiskit_target(all_instructions_target)
    assert target.instruction_supported('cx', (1, 0))


def test_cx_with_a_null_connectivity_runs_on_every_pair(
    all_instructions_target: Dict,
) -> None:
    target = ab_target_to_qiskit_target(
        {**all_instructions_target, 'connectivity': None}
    )
    assert target.instruction_supported('cx', (1, 0))


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_h_reaches_qir_as_sx(
    helium_target: Dict, requests_mock: Mocker, level: int
) -> None:
    requests_mock.register_uri('GET', '/v1/targets/', json=[helium_target])
    backend = AliceBobRemoteProvider(api_key='foo').get_backend(
        'EMU:18Q:HELIUM'
    )
    c = QuantumCircuit(1, 1)
    c.h(0)
    c.measure(0, 0)
    transpiled = transpile(
        c, backend, optimization_level=level, layout_method='trivial'
    )
    qir = circuit_to_qir(transpiled)
    assert '__quantum__qis__sx__body' in qir
    assert '__quantum__qis__h__body' not in qir
    assert '__quantum__qis__s__' not in qir


@pytest.mark.parametrize('basis', ['prepare_x', 'prepare_y'])
def test_announced_preparation_basis_is_in_target(
    all_instructions_target: Dict, basis: str
) -> None:
    assert basis in ab_target_to_qiskit_target(all_instructions_target)


@pytest.mark.parametrize('basis', ['prepare_x', 'prepare_y'])
def test_unannounced_preparation_basis_is_not_in_target(
    helium_target: Dict, basis: str
) -> None:
    assert basis not in ab_target_to_qiskit_target(helium_target)


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_preparation_marker_never_reaches_the_circuit(
    all_instructions_target: Dict, requests_mock: Mocker, level: int
) -> None:
    requests_mock.register_uri(
        'GET', '/v1/targets/', json=[all_instructions_target]
    )
    backend = AliceBobRemoteProvider(api_key='foo').get_backend(
        all_instructions_target['name']
    )
    c = QuantumCircuit(2, 2)
    c.initialize('+', 0)
    c.initialize('r', 1)
    c.cx(0, 1)
    c.measure([0, 1], [0, 1])
    transpiled = transpile(c, backend, optimization_level=level)
    assert not {'prepare_x', 'prepare_y'} & set(transpiled.count_ops())


@pytest.mark.parametrize('state', ['+', '-', 'r', 'l'])
def test_xy_preparation_gives_the_requested_state(
    helium_target: Dict, state: str
) -> None:
    c = QuantumCircuit(1)
    c.initialize(state, 0)
    lowered = _lower(helium_target, c)
    assert lowered.data[0].operation.params[0] in {'0', '1'}
    assert _evolve_after_preparation(lowered).equiv(
        Statevector.from_label(state)
    )


@pytest.mark.parametrize(
    'basis, state, bit',
    [
        ('x', '+', '0'),
        ('x', '-', '1'),
        ('y', 'r', '0'),
        ('y', 'l', '1'),
    ],
)
def test_xy_measurement_reads_the_basis_eigenstate(
    helium_target: Dict, basis: str, state: str, bit: str
) -> None:
    c = QuantumCircuit(1, 1)
    getattr(c, f'measure_{basis}')(0, 0)
    *rotations, measure = _lower(helium_target, c).data
    assert measure.operation.name == 'measure'
    rotated = Statevector.from_label(state)
    for instruction in rotations:
        rotated = rotated.evolve(instruction.operation)
    assert rotated.probabilities_dict() == pytest.approx({bit: 1.0})


@pytest.mark.parametrize('level', [0, 1, 2, 3])
@pytest.mark.parametrize(
    'operation', ['+', '-', 'r', 'l', 'measure_x', 'measure_y']
)
def test_xy_basis_costs_one_q(
    helium_target: Dict, requests_mock: Mocker, operation: str, level: int
) -> None:
    c = QuantumCircuit(1, 1)
    if operation.startswith('measure'):
        getattr(c, operation)(0, 0)
    else:
        c.initialize(operation, 0)
        c.measure(0, 0)
    transpiled = transpile(
        c,
        _helium_backend(helium_target, requests_mock),
        optimization_level=level,
    )
    ops = transpiled.count_ops()
    assert ops.get('sx', 0) == 1
    assert set(ops) <= {'initialize', 'sx', 'rz', 'x', 'measure'}
    calls = re.findall(
        r'call \w+ @__quantum__qis__(\w+?)__body', circuit_to_qir(transpiled)
    )
    assert set(calls) <= {'sx', 'rz', 'x', 'cnot', 'prepare_z', 'mz'}


@pytest.mark.parametrize('basis, state', [('x', '+'), ('y', 'r')])
def test_announced_xy_basis_is_kept(
    helium_target: Dict, requests_mock: Mocker, basis: str, state: str
) -> None:
    helium_target['instructions'] += [
        {
            'signature': f'__quantum__qis__prepare_{basis}__body:void (i1, %Qubit*)'
        },
        {
            'signature': f'__quantum__qis__m{basis}__body:void (%Qubit*, %Result*)'
        },
    ]
    c = QuantumCircuit(1, 1)
    c.initialize(state, 0)
    getattr(c, f'measure_{basis}')(0, 0)
    lowered = _lower(helium_target, c)
    assert [i.operation.name for i in lowered.data] == [
        'initialize',
        f'measure_{basis}',
    ]
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )
    calls = re.findall(
        r'call \w+ @__quantum__qis__(\w+?)__body', circuit_to_qir(transpiled)
    )
    assert calls == [f'prepare_{basis}', f'm{basis}']


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_preparation_marker_is_not_a_synthesis_gate(
    h_t_target: Dict, requests_mock: Mocker, level: int
) -> None:
    h_t_target['instructions'] += [
        {'signature': '__quantum__qis__prepare_z__body:void (i1, %Qubit*)'},
        {'signature': '__quantum__qis__prepare_x__body:void (i1, %Qubit*)'},
        {'signature': '__quantum__qis__mz__body:void (%Qubit*, %Result*)'},
    ]
    c = QuantumCircuit(1, 1)
    c.rx(0.7, 0)
    c.measure(0, 0)
    transpiled = transpile(
        c, _helium_backend(h_t_target, requests_mock), optimization_level=level
    )
    assert set(transpiled.count_ops()) <= {'initialize', 'h', 't', 'measure'}


def _reversal_warnings(
    helium_target: Dict,
    requests_mock: Mocker,
    circuit: QuantumCircuit,
    level: int,
) -> List[str]:
    backend = _helium_backend(helium_target, requests_mock)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        transpile(
            circuit,
            backend,
            optimization_level=level,
            layout_method='trivial',
        )
    return [
        str(w.message) for w in caught if 'Reversed the cx' in str(w.message)
    ]


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_reversed_cx_warns_once_with_each_pair(
    helium_target: Dict, requests_mock: Mocker, level: int
) -> None:
    c = QuantumCircuit(4)
    c.cx(1, 0)
    c.cx(2, 1)
    c.cx(1, 0)
    c.cx(3, 2)
    assert _reversal_warnings(helium_target, requests_mock, c, level) == [
        'Reversed the cx on physical qubits (1, 0) x2, (3, 2) to match the '
        'wiring.'
    ]


def _q_warnings(caught: List[warnings.WarningMessage]) -> List[str]:
    return [str(w.message) for w in caught if 'Q gate' in str(w.message)]


@pytest.mark.parametrize(
    'level, message',
    [
        (
            0,
            'The transpiled circuit holds 2 Q gates (sx). A Q gate breaks '
            'the noise bias of a cat qubit.',
        ),
        (1, None),
        (2, None),
        (3, None),
    ],
)
def test_transpile_warns_with_the_final_count_of_q_gates(
    helium_target: Dict,
    requests_mock: Mocker,
    level: int,
    message: Optional[str],
) -> None:
    c = QuantumCircuit(1)
    c.sx(0)
    c.sx(0)
    backend = _helium_backend(helium_target, requests_mock)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        transpile(c, backend, optimization_level=level)
    assert _q_warnings(caught) == ([message] if message else [])


def test_transpile_warns_about_one_q_gate(
    helium_target: Dict, requests_mock: Mocker
) -> None:
    c = QuantumCircuit(1)
    c.h(0)
    backend = _helium_backend(helium_target, requests_mock)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        transpile(c, backend, optimization_level=0)
    assert _q_warnings(caught) == [
        'The transpiled circuit holds 1 Q gate (sx). A Q gate breaks the '
        'noise bias of a cat qubit.'
    ]


def test_run_warns_with_the_count_of_q_gates(
    helium_target: Dict, successful_job: Mocker
) -> None:
    successful_job.register_uri('GET', '/v1/targets/', json=[helium_target])
    backend = AliceBobRemoteProvider(api_key='foo').get_backend(
        helium_target['name']
    )
    c = QuantumCircuit(1, 1)
    c.initialize('0', 0)
    c.sx(0)
    c.sx(0)
    c.measure(0, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        backend.run(c)
    assert _q_warnings(caught) == [
        'The circuit sent to EMU:18Q:HELIUM holds 2 Q gates (sx). A Q gate '
        'breaks the noise bias of a cat qubit.'
    ]


def test_run_counts_the_q_gates_inside_a_block(
    helium_target: Dict, successful_job: Mocker
) -> None:
    successful_job.register_uri('GET', '/v1/targets/', json=[helium_target])
    backend = AliceBobRemoteProvider(api_key='foo').get_backend(
        helium_target['name']
    )
    c = QuantumCircuit(1, 1)
    c.initialize('0', 0)
    c.measure(0, 0)
    with c.if_test((c.clbits[0], 1)):
        c.sx(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        backend.run(c)
    assert _q_warnings(caught) == [
        'The circuit sent to EMU:18Q:HELIUM holds 1 Q gate (sx). A Q gate '
        'breaks the noise bias of a cat qubit.'
    ]


def test_run_gives_no_q_warning_on_a_target_without_q(
    successful_job: Mocker,
) -> None:
    backend = AliceBobRemoteProvider(api_key='foo').get_backend('H')
    c = QuantumCircuit(1, 1)
    c.sx(0)
    c.measure(0, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        backend.run(c)
    assert _q_warnings(caught) == []


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_wired_cx_gives_no_reversal_warning(
    helium_target: Dict, requests_mock: Mocker, level: int
) -> None:
    c = QuantumCircuit(2)
    c.cx(0, 1)
    assert _reversal_warnings(helium_target, requests_mock, c, level) == []


def _ring_backend(helium_target: Dict, requests_mock: Mocker):
    """A Helium backend cut to a directed ring of 4 qubits, small enough to
    compute the unitary of a transpiled circuit."""
    ring = {
        **helium_target,
        'numQubits': 4,
        'connectivity': [
            [even, (even + step) % 4] for even in (0, 2) for step in (1, -1)
        ],
    }
    return _helium_backend(ring, requests_mock)


def _on_qubit(circuit: QuantumCircuit, qubit: int) -> List[Any]:
    """The instructions that act on one qubit, in circuit order."""
    return [
        instruction
        for instruction in circuit.data
        if qubit in {circuit.find_bit(q).index for q in instruction.qubits}
    ]


def _wire(circuit: QuantumCircuit, qubit: int) -> List[Tuple[Any, ...]]:
    """The gates on one qubit, in circuit order, without preparations and
    measurements."""
    return [
        (instruction.name, *instruction.operation.params)
        for instruction in _on_qubit(circuit, qubit)
        if instruction.name not in {'initialize', 'measure'}
    ]


def _cx_pairs(circuit: QuantumCircuit) -> List[Tuple[int, ...]]:
    return [
        tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        for instruction in circuit.data
        if instruction.operation.name == 'cx'
    ]


def _rz(angle: float) -> Tuple[Any, ...]:
    return ('rz', pytest.approx(angle))


_SX = ('sx',)
_X = ('x',)
_CX = ('cx',)
_H = [_rz(pi / 2), _SX, _rz(pi / 2)]


@pytest.mark.parametrize(
    'gate, form',
    [
        ('h', _H),
        ('s', [_rz(pi / 2)]),
        ('sdg', [_rz(-pi / 2)]),
        ('t', [_rz(pi / 4)]),
        ('tdg', [_rz(-pi / 4)]),
        ('z', [_rz(pi)]),
        ('y', [_rz(pi), _X]),
    ],
)
def test_single_qubit_gate_has_its_native_form_at_level_0(
    helium_target: Dict,
    requests_mock: Mocker,
    gate: str,
    form: List[Tuple[Any, ...]],
) -> None:
    c = QuantumCircuit(1)
    getattr(c, gate)(0)
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )
    assert _wire(transpiled, 0) == form


_Q_DAGGER = [_rz(pi), _SX, _rz(pi)]


@pytest.mark.parametrize(
    'build, form',
    [
        (lambda c: c.rx(pi / 2, 0), [_SX]),
        (lambda c: c.rx(-pi / 2, 0), _Q_DAGGER),
        (lambda c: c.rx(pi, 0), [_X]),
        (
            lambda c: c.rx(0.3, 0),
            [_rz(pi / 2), _SX, _rz(0.3 - pi), _SX, _rz(pi / 2)],
        ),
        (lambda c: c.sxdg(0), _Q_DAGGER),
        (lambda c: c.ry(0.3, 0), [_SX, _rz(0.3 - pi), _SX, _rz(pi)]),
        (lambda c: c.ry(pi / 2, 0), [_rz(-pi / 2), _SX, _rz(pi / 2)]),
        (lambda c: c.ry(-pi / 2, 0), [_rz(pi / 2), _SX, _rz(-pi / 2)]),
        (lambda c: c.ry(pi, 0), [_X, _rz(pi)]),
        (lambda c: c.rx(3 * pi, 0), [_X]),
        (lambda c: c.rx(-pi, 0), [_X]),
        (lambda c: c.rx(5 * pi / 2, 0), [_SX]),
        (lambda c: c.rx(-3 * pi / 2, 0), [_SX]),
        (lambda c: c.ry(-pi, 0), [_X, _rz(pi)]),
        (lambda c: c.ry(5 * pi / 2, 0), [_rz(-pi / 2), _SX, _rz(pi / 2)]),
        (
            lambda c: c.rx(0.3 + 2 * pi, 0),
            [_rz(pi / 2), _SX, _rz(0.3 - pi), _SX, _rz(pi / 2)],
        ),
        (lambda c: c.rx(pi + 1e-12, 0), [_X]),
        (
            lambda c: c.rx(pi + 2e-5, 0),
            [_rz(pi / 2), _SX, _rz(-2 * pi + 2e-5), _SX, _rz(pi / 2)],
        ),
        (lambda c: c.rx(0, 0), []),
        (lambda c: c.ry(0, 0), []),
        (lambda c: c.rx(2 * pi, 0), []),
        (lambda c: c.ry(-4 * pi, 0), []),
        (lambda c: c.rx(1e-12, 0), []),
    ],
    ids=[
        'rx(pi/2)',
        'rx(-pi/2)',
        'rx(pi)',
        'rx(0.3)',
        'sxdg',
        'ry(0.3)',
        'ry(pi/2)',
        'ry(-pi/2)',
        'ry(pi)',
        'rx(3pi)',
        'rx(-pi)',
        'rx(5pi/2)',
        'rx(-3pi/2)',
        'ry(-pi)',
        'ry(5pi/2)',
        'rx(0.3+2pi)',
        'rx(pi+1e-12)',
        'rx(pi+2e-5)',
        'rx(0)',
        'ry(0)',
        'rx(2pi)',
        'ry(-4pi)',
        'rx(1e-12)',
    ],
)
def test_x_and_y_rotations_have_their_native_form_at_level_0(
    helium_target: Dict,
    requests_mock: Mocker,
    build: Callable[[QuantumCircuit], Any],
    form: List[Tuple[Any, ...]],
) -> None:
    c = QuantumCircuit(4)
    build(c)
    transpiled = transpile(
        c, _ring_backend(helium_target, requests_mock), optimization_level=0
    )
    assert _wire(transpiled, 0) == form
    assert _unitary(transpiled) == Operator(c)


def test_x_rotation_is_kept_on_a_target_that_has_it(
    helium_target: Dict, requests_mock: Mocker
) -> None:
    helium_target['instructions'].append(
        {'signature': '__quantum__qis__rx__body:void (double, %Qubit*)'}
    )
    c = QuantumCircuit(1)
    c.rx(0.3, 0)
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )
    assert _wire(transpiled, 0) == [('rx', pytest.approx(0.3))]


def test_y_rotation_is_kept_on_a_target_that_has_it(
    helium_target: Dict, requests_mock: Mocker
) -> None:
    helium_target['instructions'].append(
        {'signature': '__quantum__qis__ry__body:void (double, %Qubit*)'}
    )
    c = QuantumCircuit(1)
    c.ry(0.3, 0)
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )
    assert _wire(transpiled, 0) == [('ry', pytest.approx(0.3))]


@pytest.mark.parametrize(
    'gate, qubits, wire_0, wire_1',
    [
        ('cx', (1, 0), _H + [_CX] + _H, _H + [_CX] + _H),
        ('cz', (0, 1), [_CX], _H + [_CX] + _H),
        (
            'swap',
            (0, 1),
            [_CX] + _H + [_CX] + _H + [_CX],
            [_CX] + _H + [_CX] + _H + [_CX],
        ),
    ],
)
def test_two_qubit_gate_has_its_native_form_at_level_0(
    helium_target: Dict,
    requests_mock: Mocker,
    gate: str,
    qubits: Tuple[int, int],
    wire_0: List[Tuple[Any, ...]],
    wire_1: List[Tuple[Any, ...]],
) -> None:
    c = QuantumCircuit(2)
    getattr(c, gate)(*qubits)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        transpiled = transpile(
            c,
            _helium_backend(helium_target, requests_mock),
            optimization_level=0,
        )
    assert _wire(transpiled, 0) == wire_0
    assert _wire(transpiled, 1) == wire_1
    assert set(_cx_pairs(transpiled)) == {(0, 1)}


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_reset_becomes_a_preparation_of_zero(
    helium_target: Dict, requests_mock: Mocker, level: int
) -> None:
    c = QuantumCircuit(1, 1)
    c.x(0)
    c.reset(0)
    c.measure(0, 0)
    transpiled = transpile(
        c,
        _helium_backend(helium_target, requests_mock),
        optimization_level=level,
    )
    assert 'reset' not in transpiled.count_ops()
    preparations = [
        i.operation.params for i in transpiled.data if i.name == 'initialize'
    ]
    assert preparations[-1] == ['0']


def test_native_circuit_is_kept_at_level_0(
    helium_target: Dict, requests_mock: Mocker
) -> None:
    c = QuantumCircuit(2, 1)
    c.initialize('0', 0)
    c.initialize('1', 1)
    c.x(0)
    c.sx(0)
    c.rz(0.3, 1)
    c.cx(0, 1)
    c.delay(16, 0, unit='ns')
    c.measure(1, 0)
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )

    def operations(circuit: QuantumCircuit, qubit: int) -> List[Tuple]:
        return [
            (i.name, *(circuit.find_bit(q).index for q in i.qubits))
            for i in _on_qubit(circuit, qubit)
        ]

    for qubit in (0, 1):
        assert operations(transpiled, qubit) == operations(c, qubit)


def test_initialize_splits_one_qubit_at_a_time(
    helium_target: Dict, requests_mock: Mocker
) -> None:
    helium_target['instructions'].append(
        {'signature': '__quantum__qis__prepare_x__body:void (i1, %Qubit*)'}
    )
    c = QuantumCircuit(3)
    c.initialize('+-0', [0, 1, 2])
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )
    assert sorted(
        (transpiled.find_bit(i.qubits[0]).index, i.operation.params)
        for i in transpiled.data
        if i.name == 'initialize'
    ) == [(0, ['0']), (1, ['-']), (2, ['+'])]


def _unitary(circuit: QuantumCircuit) -> Operator:
    gates = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name != 'initialize':
            gates.append(instruction)
    return Operator.from_circuit(gates)


_GATES: List[Tuple[str, Callable[[QuantumCircuit], Any]]] = [
    ('h', lambda c: c.h(0)),
    ('s', lambda c: c.s(0)),
    ('sdg', lambda c: c.sdg(0)),
    ('t', lambda c: c.t(0)),
    ('tdg', lambda c: c.tdg(0)),
    ('x', lambda c: c.x(0)),
    ('y', lambda c: c.y(0)),
    ('z', lambda c: c.z(0)),
    ('sx', lambda c: c.sx(0)),
    ('sxdg', lambda c: c.sxdg(0)),
    ('rx(0.3)', lambda c: c.rx(0.3, 0)),
    ('rx(pi/2)', lambda c: c.rx(pi / 2, 0)),
    ('rx(-pi/2)', lambda c: c.rx(-pi / 2, 0)),
    ('rx(pi)', lambda c: c.rx(pi, 0)),
    ('ry(0.3)', lambda c: c.ry(0.3, 0)),
    ('rz(0.3)', lambda c: c.rz(0.3, 0)),
    ('cx(1,0)', lambda c: c.cx(1, 0)),
    ('cx(1,3)', lambda c: c.cx(1, 3)),
    ('cz(0,1)', lambda c: c.cz(0, 1)),
    ('cz(1,0)', lambda c: c.cz(1, 0)),
    ('swap(1,0)', lambda c: c.swap(1, 0)),
]


@pytest.mark.parametrize('level', [0, 1, 2, 3])
@pytest.mark.parametrize(
    'build', [build for _, build in _GATES], ids=[name for name, _ in _GATES]
)
def test_translation_keeps_the_unitary(
    helium_target: Dict,
    requests_mock: Mocker,
    build: Callable[[QuantumCircuit], Any],
    level: int,
) -> None:
    c = QuantumCircuit(4)
    build(c)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        transpiled = transpile(
            c,
            _ring_backend(helium_target, requests_mock),
            optimization_level=level,
        )
    assert _unitary(transpiled).equiv(Operator(c))


def test_rotation_inside_a_custom_gate_has_its_native_form(
    helium_target: Dict, requests_mock: Mocker
) -> None:
    custom = QuantumCircuit(1, name='custom')
    custom.rx(pi / 2, 0)
    c = QuantumCircuit(1)
    c.append(custom.to_gate(), [0])
    transpiled = transpile(
        c, _helium_backend(helium_target, requests_mock), optimization_level=0
    )
    assert _wire(transpiled, 0) == [_SX]
