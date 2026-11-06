import base64
import json
import socket
import subprocess
import sys
from threading import Thread
from time import monotonic, sleep
from typing import Callable, Dict, List, Optional

import pytest

pytest.importorskip('qat')

from qat.comm.exceptions.ttypes import ErrorType, QPUException
from qat.core import HardwareSpecs, Job, Result
from qat.core.plugins import AbstractPlugin
from qat.core.qpu import QPUHandler
from qat.qpus import RemoteQPU
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.providers import JobStatus
from qiskit.providers.exceptions import (
    QiskitBackendNotFoundError,
)
from qiskit.result import Result as QiskitResult
from qiskit.transpiler import PassManager
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.plugins.state_preparation import (
    EnsurePreparationPass,
)
from qiskit_alice_bob_provider.qaptiva import (
    AliceBobQaptivaBackend,
    AliceBobQaptivaProvider,
)
from qiskit_alice_bob_provider.qir import circuit_to_qir
from qiskit_alice_bob_provider.remote.api.client import ApiClient
from qiskit_alice_bob_provider.remote.backend import (
    AliceBobRemoteBackend,
)
from qiskit_alice_bob_provider.remote.job import AliceBobRemoteJob
from qiskit_alice_bob_provider.remote.provider import (
    AliceBobRemoteProvider,
)

# The histogram of the `successful_job` fixture of the remote tests.
HISTOGRAM = {'11': 12, '10': 474, '01': 6, '00': 508}


class _FakeServer(QPUHandler):
    """A myQLM QPU that answers as the anb-myqlm-server does, and keeps the
    jobs it receives."""

    def __init__(self, targets: List[Dict]) -> None:
        super().__init__()
        self.targets = targets
        self.jobs: List[Job] = []
        self.protocol_version: Optional[str] = '1'
        self.error: Optional[QPUException] = None

    def submit_job(self, job: Job) -> Result:
        if self.error is not None:
            raise self.error
        self.jobs.append(job)
        result = Result(lsb_first=True, nbqbits=job.circuit.nbqbits)
        result.meta_data = {
            'counts': json.dumps(HISTOGRAM),
            'job_id': 'server-job',
        }
        return result

    def get_specs(self) -> HardwareSpecs:
        if self.error is not None:
            raise self.error
        meta_data = {'targets': json.dumps(self.targets)}
        if self.protocol_version is not None:
            meta_data['protocol_version'] = self.protocol_version
        return HardwareSpecs(nbqbits=1, meta_data=meta_data)


class _IdentityPlugin(AbstractPlugin):
    """A myQLM plugin that changes nothing, to stack in front of a QPU."""

    def compile(self, batch, specs):
        return batch

    def do_post_processing(self):
        return False


def _abort(message: str) -> QPUException:
    return QPUException(
        code=ErrorType.ABORT, modulename='alice_bob_qpu', message=message
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.5):
                return
        except OSError:
            sleep(0.1)
    raise RuntimeError(f'Server on port {port} did not start')


@pytest.fixture
def serve() -> Callable[[List[Dict]], _FakeServer]:
    """Serve a fake server and return it with a RemoteQPU connected to it.

    The server runs on a daemon thread, so no teardown is needed.
    """

    def _serve(targets: List[Dict]) -> _FakeServer:
        server = _FakeServer(targets)
        port = _free_port()
        Thread(
            target=server.serve,
            kwargs={'port': port, 'host_ip': '127.0.0.1'},
            daemon=True,
        ).start()
        _wait_for_port(port)
        server.remote = RemoteQPU(port=port, ip='127.0.0.1')
        return server

    return _serve


@pytest.fixture
def server(serve, targets: List[Dict]) -> _FakeServer:
    return serve(targets)


@pytest.fixture
def provider(server: _FakeServer) -> AliceBobQaptivaProvider:
    return AliceBobQaptivaProvider(server.remote)


def _circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 2)
    circuit.initialize('+', 0)
    circuit.measure_x(0, 0)
    circuit.measure(0, 1)
    return circuit


def test_package_import_does_not_import_qat() -> None:
    code = (
        'import sys, qiskit_alice_bob_provider; '
        "assert 'qat' not in sys.modules"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_backend_matches_remote_backend(targets: List[Dict]) -> None:
    for target in targets:
        qaptiva = AliceBobQaptivaBackend(None, target)
        remote = AliceBobRemoteBackend(
            ApiClient(api_key='foo', url='http://localhost'), target
        )
        assert set(qaptiva.target.operation_names) == set(
            remote.target.operation_names
        )
        assert qaptiva.options.__dict__ == remote.options.__dict__
        assert (
            qaptiva.get_translation_stage_plugin()
            == remote.get_translation_stage_plugin()
        )


def test_backends_list_the_server_targets(
    provider: AliceBobQaptivaProvider, targets: List[Dict]
) -> None:
    assert sorted(b.name for b in provider.backends()) == sorted(
        t['name'] for t in targets
    )


def test_unknown_backend_raises(provider: AliceBobQaptivaProvider) -> None:
    with pytest.raises(QiskitBackendNotFoundError):
        provider.get_backend('EMU:1Q:UNKNOWN')


def test_provider_refuses_an_object_that_is_not_a_qpu() -> None:
    with pytest.raises(TypeError, match='myQLM QPU'):
        AliceBobQaptivaProvider('10.0.0.1:1235')


def test_provider_accepts_a_plugin_stack(server: _FakeServer) -> None:
    provider = AliceBobQaptivaProvider(_IdentityPlugin() | server.remote)
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    backend.run(transpile(_circuit(), backend), shots=10)
    assert server.jobs[-1].meta_data['backend'] == 'EMU:1Q:LESCANNE_2020'


@pytest.mark.parametrize('version', [None, '2'])
def test_provider_refuses_another_protocol_version(
    server: _FakeServer, version: Optional[str]
) -> None:
    server.protocol_version = version
    with pytest.raises(QiskitError, match='protocol version'):
        AliceBobQaptivaProvider(server.remote)


def test_server_error_on_specs_raises_a_qiskit_error(
    server: _FakeServer,
) -> None:
    server.error = _abort('Could not retrieve API targets: down')
    with pytest.raises(QiskitError, match='Could not retrieve API targets'):
        AliceBobQaptivaProvider(server.remote)


def test_server_error_on_run_raises_a_qiskit_error(
    provider: AliceBobQaptivaProvider, server: _FakeServer
) -> None:
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    server.error = _abort('API job failed: boom')
    with pytest.raises(QiskitError, match='API job failed: boom'):
        backend.run(transpile(_circuit(), backend))


def test_get_backend_accepts_verbose(
    provider: AliceBobQaptivaProvider,
) -> None:
    """A call to the remote provider works on Qaptiva."""
    provider.get_backend('EMU:1Q:LESCANNE_2020', verbose=False)


def test_get_backend_sets_options(provider: AliceBobQaptivaProvider) -> None:
    backend = provider.get_backend(
        'EMU:1Q:LESCANNE_2020', average_nb_photons=6.0
    )
    assert backend.options.average_nb_photons == 6.0


@pytest.mark.parametrize(
    'options', [{'foo': 1}, {'average_nb_photons': 100.0}, {'shots': 0}]
)
def test_invalid_options_raise(
    provider: AliceBobQaptivaProvider, options: Dict
) -> None:
    with pytest.raises(ValueError):
        provider.get_backend('EMU:1Q:LESCANNE_2020', **options)
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    with pytest.raises(ValueError):
        backend.run(transpile(_circuit(), backend), **options)


def test_run_sends_qir_backend_and_input_params(
    provider: AliceBobQaptivaProvider, server: _FakeServer
) -> None:
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    circuit = transpile(_circuit(), backend)
    backend.run(circuit, shots=200, average_nb_photons=5.0)

    job = server.jobs[-1]
    expected_qir = circuit_to_qir(
        PassManager([EnsurePreparationPass()]).run(circuit)
    )
    assert base64.b64decode(job.meta_data['qir']).decode() == expected_qir
    assert job.meta_data['backend'] == 'EMU:1Q:LESCANNE_2020'
    assert json.loads(job.meta_data['input_params']) == {
        'nbShots': 200,
        'averageNbPhotons': 5.0,
    }
    assert job.nbshots == 200
    assert backend.options.shots == 1000


def test_result_matches_remote_result(
    provider: AliceBobQaptivaProvider, successful_job: Mocker
) -> None:
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    circuit = transpile(_circuit(), backend)
    job = backend.run(circuit, shots=100)
    result = job.result()

    remote: AliceBobRemoteBackend = AliceBobRemoteProvider(
        api_key='foo'
    ).get_backend('EMU:1Q:LESCANNE_2020', verbose=False)
    remote_job: AliceBobRemoteJob = remote.run(circuit, shots=100)
    remote_result = remote_job.result(wait=0)

    assert isinstance(result, QiskitResult)
    assert result.get_counts() == remote_result.get_counts() == HISTOGRAM
    assert job.job_id() == 'server-job'
    assert job.status() == JobStatus.DONE


def test_memory_raises(serve, single_cat_target: Dict) -> None:
    single_cat_target['inputParams']['memory'] = {
        'required': False,
        'default': False,
        'constraints': [],
    }
    server = serve([single_cat_target])
    backend = AliceBobQaptivaProvider(server.remote).get_backend(
        'EMU:1Q:LESCANNE_2020'
    )
    with pytest.raises(ValueError, match='memory'):
        backend.run(transpile(_circuit(), backend), memory=True)
    assert server.jobs == []
