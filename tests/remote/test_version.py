from mock import MagicMock, patch
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.remote.api.version import (
    PROVIDER_PYPI_URL,
    ProviderStatus,
    get_provider_status,
)


@patch(
    'qiskit_alice_bob_provider.remote.api.version.version',
    wraps=lambda _: '1.2.0',
)
def test_get_provider_status_latest(
    version_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    requests_mock.register_uri(
        'GET',
        PROVIDER_PYPI_URL,
        json={
            'info': {
                'author': 'Alice & Bob Software Team',
                'keywords': 'Qiskit, Alice & Bob, Quantum, SDK',
                'license': 'Apache 2.0',
                'name': 'qiskit-alice-bob-provider',
                'summary': (
                    'Provider for running Qiskit circuits on '
                    'Alice & Bob QPUs and simulators'
                ),
                'version': '1.2.0',
            },
        },
    )

    status = get_provider_status()
    version_mock.assert_called_once()
    assert status == ProviderStatus.LATEST


@patch(
    'qiskit_alice_bob_provider.remote.api.version.version',
    wraps=lambda _: '1.2.0',
)
def test_get_provider_status_outdated(
    version_mock: MagicMock,
    requests_mock: Mocker,
) -> None:
    requests_mock.register_uri(
        'GET',
        PROVIDER_PYPI_URL,
        json={
            'info': {
                'author': 'Alice & Bob Software Team',
                'keywords': 'Qiskit, Alice & Bob, Quantum, SDK',
                'license': 'Apache 2.0',
                'name': 'qiskit-alice-bob-provider',
                'summary': (
                    'Provider for running Qiskit circuits on '
                    'Alice & Bob QPUs and simulators'
                ),
                'version': '1.2.1',
            },
        },
    )

    status = get_provider_status()
    version_mock.assert_called_once()
    assert status == ProviderStatus.OUTDATED


def test_get_provider_status_pypi_exception(
    requests_mock: Mocker,
) -> None:
    requests_mock.register_uri(
        'GET', PROVIDER_PYPI_URL, json={}, status_code=404
    )

    assert get_provider_status() == ProviderStatus.UNKNOWN
