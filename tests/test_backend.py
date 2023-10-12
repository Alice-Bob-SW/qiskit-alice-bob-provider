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

# pylint: disable=unused-argument

import pytest
from qiskit import QiskitError, QuantumCircuit, execute
from qiskit.providers import Options
from qiskit.result import Result
from qiskit.transpiler.exceptions import TranspilerError
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.remote.api.client import AliceBobApiException
from qiskit_alice_bob_provider.remote.backend import (
    _ab_input_params_from_options,
    _qiskit_to_qir,
)
from qiskit_alice_bob_provider.remote.provider import AliceBobRemoteProvider


def test_options_validation(mocked_targets) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(ValueError):
        execute(c, backend, average_nb_photons=40)
    with pytest.raises(ValueError):
        execute(c, backend, average_nb_photons=-1)
    with pytest.raises(ValueError):
        execute(c, backend, bad_option=1)
    with pytest.raises(ValueError):
        execute(c, backend, shots=0)
    with pytest.raises(ValueError):
        execute(c, backend, shots=1e10)


def test_too_many_qubits_clients_side(mocked_targets) -> None:
    c = QuantumCircuit(3, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(TranspilerError):
        execute(c, backend)


def test_counts_ordering(successful_job: Mocker) -> None:
    c = QuantumCircuit(1, 2)
    c.initialize('+', 0)
    c.measure_x(0, 0)
    c.measure(0, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = execute(c, backend)
    counts = job.result(wait=0).get_counts()
    expected = {'11': 12, '10': 474, '01': 6, '00': 508}
    assert counts == expected
    counts = job.result(wait=0).get_counts()  # testing memoization
    assert counts == expected


def test_failed_transpilation(failed_transpilation_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = execute(c, backend)
    res: Result = job.result(wait=0)
    assert res.results[0].data.input_qir is not None
    assert res.results[0].data.transpiled_qir is None
    res = job.result(wait=0)  # testing memoization
    assert res.results[0].data.input_qir is not None
    assert res.results[0].data.transpiled_qir is None
    with pytest.raises(QiskitError):
        res.get_counts()


def test_failed_execution(failed_execution_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = execute(c, backend)
    res: Result = job.result(wait=0)
    assert res.results[0].data.input_qir is not None
    assert res.results[0].data.transpiled_qir is not None
    with pytest.raises(QiskitError):
        res.get_counts()


def test_cancel_job(cancellable_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = execute(c, backend)
    job.cancel()
    res: Result = job.result(wait=0)
    assert res.results[0].data.input_qir is not None
    assert res.results[0].data.transpiled_qir is not None
    with pytest.raises(QiskitError):
        res.get_counts()


def test_failed_server_side_validation(failed_validation_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(AliceBobApiException):
        execute(c, backend)


def test_delay_instruction_recognized() -> None:
    c = QuantumCircuit(1, 2)
    c.initialize('+', 0)
    c.measure_x(0, 0)
    c.delay(3000, 0, unit='ns')
    c.measure(0, 1)
    qir = _qiskit_to_qir(c)
    delay_call = (
        'call void @__quantum__qis__delay__body'
        '(double 3.000000e+00, %Qubit* null)'
    )
    assert delay_call in qir


def test_ab_input_params_from_options() -> None:
    options = Options(shots=43, average_nb_photons=3.2, foo_hey='bar')
    params = _ab_input_params_from_options(options)
    assert params == {'nbShots': 43, 'averageNbPhotons': 3.2, 'fooHey': 'bar'}
