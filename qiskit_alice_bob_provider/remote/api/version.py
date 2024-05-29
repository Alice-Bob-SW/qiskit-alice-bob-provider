import logging
from enum import Enum

import requests
from pkg_resources import get_distribution

PROVIDER_PYPI_URL = 'https://pypi.org/pypi/qiskit-alice-bob-provider/json'


class ProviderStatus(Enum):
    LATEST = 'LATEST'
    OUTDATED = 'OUTDATED'
    UNKNOWN = 'UNKNOWN'


def get_provider_status() -> ProviderStatus:
    """Query the Pypi API to compare the latest release of the Qiskit provider
    with the current installation and return the update status
    (latest, outdated, unknown)."""
    try:
        pypi_response = requests.get(url=PROVIDER_PYPI_URL, timeout=1.0)
        assert pypi_response.status_code == 200
        pypi_version = pypi_response.json()['info']['version']
    # pylint: disable=broad-exception-caught
    except Exception as e:
        logging.exception(e)
        return ProviderStatus.UNKNOWN

    installed_version = get_distribution('qiskit_alice_bob_provider').version

    if pypi_version == installed_version:
        return ProviderStatus.LATEST
    return ProviderStatus.OUTDATED
