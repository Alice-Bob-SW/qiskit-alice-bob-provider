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

import warnings
from typing import Dict, Tuple

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import SXGate
from qiskit.quantum_info import Operator
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.qir import circuit_to_qir
from qiskit_alice_bob_provider.remote.provider import AliceBobRemoteProvider
from qiskit_alice_bob_provider.remote.qir_to_qiskit import (
    ab_target_to_qiskit_target,
)


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
