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
from typing import Dict, List, Tuple

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
def test_reversed_cx_warns_once_with_each_pair_and_its_cost(
    helium_target: Dict, requests_mock: Mocker, level: int
) -> None:
    c = QuantumCircuit(4)
    c.cx(1, 0)
    c.cx(2, 1)
    c.cx(1, 0)
    c.cx(3, 2)
    assert _reversal_warnings(helium_target, requests_mock, c, level) == [
        'Reversed the cx on physical qubits (1, 0) x2, (3, 2) to match the '
        'wiring. This adds 12 Q gates before optimization, which cost bias.'
    ]


@pytest.mark.parametrize('level', [0, 1, 2, 3])
def test_wired_cx_gives_no_reversal_warning(
    helium_target: Dict, requests_mock: Mocker, level: int
) -> None:
    c = QuantumCircuit(2)
    c.cx(0, 1)
    assert _reversal_warnings(helium_target, requests_mock, c, level) == []
