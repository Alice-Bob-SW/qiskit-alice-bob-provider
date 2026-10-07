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


from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
from qiskit import QiskitError, QuantumCircuit, transpile
from qiskit.providers import Options
from qiskit.result import Result
from qiskit.transpiler.exceptions import TranspilerError
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.qir import circuit_to_qir
from qiskit_alice_bob_provider.remote.api.client import AliceBobApiException
from qiskit_alice_bob_provider.remote.backend import (
    AliceBobRemoteBackend,
    _ab_input_params_from_options,
)
from qiskit_alice_bob_provider.remote.provider import AliceBobRemoteProvider


def test_get_backend(mocked_targets) -> None:
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    assert isinstance(backend, AliceBobRemoteBackend)
    assert backend.options['average_nb_photons'] == 4.0  # Default value.


def test_get_backend_with_options(mocked_targets) -> None:
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend(
        'EMU:1Q:LESCANNE_2020', average_nb_photons=6.0
    )
    assert isinstance(backend, AliceBobRemoteBackend)
    assert backend.options['average_nb_photons'] == 6.0


def test_get_multiple_backends_with_options(mocked_targets) -> None:
    """
    Test that getting multiple backends with different options does not affect
    the default backend options.
    """
    provider = AliceBobRemoteProvider(api_key='foo')
    default_backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    _ = provider.get_backend(
        'EMU:1Q:LESCANNE_2020', average_nb_photons=6, shots=10
    )
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')

    # Ensure that the backend objects are different instances
    assert default_backend is not backend
    # Ensure that the options objects are different instances
    assert default_backend.options is not backend.options
    # Ensure that the default options are equal
    assert default_backend.options == backend.options


def test_get_backend_options_validation(mocked_targets) -> None:
    provider = AliceBobRemoteProvider(api_key='foo')
    with pytest.raises(ValueError):
        provider.get_backend('EMU:1Q:LESCANNE_2020', average_nb_photons=40)
    with pytest.raises(ValueError):
        provider.get_backend('EMU:1Q:LESCANNE_2020', average_nb_photons=-1)
    with pytest.raises(ValueError):
        provider.get_backend('EMU:1Q:LESCANNE_2020', shots=0)
    with pytest.raises(ValueError):
        provider.get_backend('EMU:1Q:LESCANNE_2020', shots=1e10)
    with pytest.raises(ValueError):
        provider.get_backend('EMU:1Q:LESCANNE_2020', bad_option=1)


def test_execute_options_validation(mocked_targets) -> None:
    # We are permissive in our options system, allowing the user to both
    # define options when creating the backend and executing.
    # We therefore need to test both behaviors.
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(ValueError):
        backend.run(c, average_nb_photons=40)
    with pytest.raises(ValueError):
        backend.run(c, average_nb_photons=-1)
    with pytest.raises(ValueError):
        backend.run(c, bad_option=1)
    with pytest.raises(ValueError):
        backend.run(c, shots=0)
    with pytest.raises(ValueError):
        backend.run(c, shots=1e10)


def test_too_many_qubits_clients_side(mocked_targets) -> None:
    c = QuantumCircuit(3, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(TranspilerError):
        transpile(c, backend)


def test_input_not_quantum_circuit(mocked_targets) -> None:
    c1 = QuantumCircuit(1, 1)
    c2 = QuantumCircuit(1, 1)
    s1 = object()
    s2 = object()
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(NotImplementedError):
        backend.run([c1, c2])
    with pytest.raises(NotImplementedError):
        backend.run(s1)
    with pytest.raises(NotImplementedError):
        backend.run([s1, s2])


def test_counts_ordering(successful_job: Mocker) -> None:
    c = QuantumCircuit(1, 2)
    c.initialize('+', 0)
    c.measure_x(0, 0)
    c.measure(0, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = backend.run(c)
    counts = job.result(wait=0).get_counts()
    expected = {'11': 12, '10': 474, '01': 6, '00': 508}
    assert counts == expected
    counts = job.result(wait=0).get_counts()  # testing memoization
    assert counts == expected


def test_failed_transpilation(failed_transpilation_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = backend.run(c)
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
    job = backend.run(c)
    res: Result = job.result(wait=0)
    assert res.results[0].data.input_qir is not None
    assert res.results[0].data.transpiled_qir is not None
    with pytest.raises(QiskitError):
        res.get_counts()


def test_cancel_job(cancellable_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = backend.run(c)
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
        backend.run(c)


def _job_created(mocker: Mocker) -> bool:
    return any(
        r.method == 'POST' and r.path == '/v1/jobs/'
        for r in mocker.request_history
    )


def test_dt_delay_refused_before_job_creation(successful_job: Mocker) -> None:
    c = QuantumCircuit(1, 1)
    c.delay(100_000, 0, unit='dt')
    c.measure(0, 0)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    transpiled = transpile(c, backend)
    with pytest.raises(ValueError, match=r'qubits \[0\] has the unit "dt"'):
        backend.run(transpiled)
    assert not _job_created(successful_job)


def test_dt_delay_in_if_test_refused_before_job_creation(
    successful_job: Mocker,
) -> None:
    c = QuantumCircuit(1, 1)
    c.measure(0, 0)
    with c.if_test((c.clbits[0], 1)):
        c.delay(100_000, 0, unit='dt')
    c.measure(0, 0)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(ValueError, match=r'qubits \[0\] has the unit "dt"'):
        backend.run(c)
    assert not _job_created(successful_job)


def test_dt_delay_in_block_reports_the_outer_qubit(
    successful_job: Mocker,
) -> None:
    c = QuantumCircuit(3, 1)
    c.measure(0, 0)
    with c.if_test((c.clbits[0], 1)):
        c.delay(100_000, 2, unit='dt')
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('ALL_INSTRUCTIONS')
    with pytest.raises(ValueError, match=r'qubits \[2\] has the unit'):
        backend.run(c)
    assert not _job_created(successful_job)


@pytest.mark.parametrize(
    'duration, unit, qir_duration',
    [(100, 'us', '1.000000e+02'), (3000, 'ns', '3.000000e+00')],
)
def test_time_unit_delay_runs(
    successful_job: Mocker, duration: int, unit: str, qir_duration: str
) -> None:
    c = QuantumCircuit(1, 1)
    c.delay(duration, 0, unit=unit)
    c.measure(0, 0)
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = backend.run(transpile(c, backend))
    assert job.result(wait=0).get_counts()
    upload = next(
        r
        for r in successful_job.request_history
        if r.method == 'POST' and r.path == '/v1/jobs/my-job/input'
    )
    delay_call = (
        'call void @__quantum__qis__delay__body'
        f'(double {qir_duration}, %Qubit* null)'
    )
    assert delay_call in upload.body.decode()


def test_ab_input_params_from_options() -> None:
    options = Options(shots=43, average_nb_photons=3.2, foo_hey='bar')
    params = _ab_input_params_from_options(options)
    assert params == {'nbShots': 43, 'averageNbPhotons': 3.2, 'fooHey': 'bar'}


def test_ab_input_params_from_options_with_numpy() -> None:
    options = Options(shots=43, average_nb_photons=np.int32(3), foo_hey='bar')
    params = _ab_input_params_from_options(options)
    assert params == {'nbShots': 43, 'averageNbPhotons': 3, 'fooHey': 'bar'}


def test_translation_plugin_and_qir(mocked_targets) -> None:
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('ALL_INSTRUCTIONS')

    c = QuantumCircuit(4, 4)
    c.initialize('-01+')
    c.measure([0, 1], [2, 3])
    c.measure_x(2, 0)
    c.measure_x(3, 1)

    transpiled = transpile(c, backend)
    qir = circuit_to_qir(transpiled)

    assert (
        dedent(
            Path(
                'tests/resources/test_translation_plugin_and_qir.ll'
            ).read_text(encoding='utf-8')
        )
        in qir
    )


def test_determine_translation_plugin(mocked_targets) -> None:
    p = AliceBobRemoteProvider(api_key='foo')

    # Rotations available
    assert (
        p.get_backend('ALL_INSTRUCTIONS').get_translation_stage_plugin()
        == 'state_preparation'
    )

    # H and T missing
    assert (
        p.get_backend('EMU:1Q:LESCANNE_2020').get_translation_stage_plugin()
        == 'state_preparation'
    )

    # T missing
    assert (
        p.get_backend('H').get_translation_stage_plugin()
        == 'state_preparation'
    )

    # ok
    assert (
        p.get_backend('H_T').get_translation_stage_plugin() == 'sk_synthesis'
    )


def test_y_basis_circuit_converts(mocked_targets) -> None:
    provider = AliceBobRemoteProvider(api_key='foo')
    backend = provider.get_backend('ALL_INSTRUCTIONS')
    c = QuantumCircuit(2, 2)
    c.initialize('lr')
    c.measure_y(0, 0)
    c.measure_y(1, 1)
    qir = circuit_to_qir(transpile(c, backend))
    assert 'call void @__quantum__qis__prepare_y__body(i1 false' in qir
    assert 'call void @__quantum__qis__prepare_y__body(i1 true' in qir
    assert qir.count('call void @__quantum__qis__my__body(') == 2
