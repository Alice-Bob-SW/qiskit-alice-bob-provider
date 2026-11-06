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

import logging
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, TypeVar

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.providers.providerutils import filter_backends

from qiskit_alice_bob_provider.remote.api.version import (
    ProviderStatus,
    get_provider_status,
)

from .api.client import ApiClient
from .api.targets import list_targets
from .backend import AliceBobRemoteBackend, AliceBobTargetBackend

BackendT = TypeVar('BackendT', bound=AliceBobTargetBackend)


class AliceBobTargetProvider(ABC, Generic[BackendT]):
    """Base class of the providers of backends built from Alice & Bob API
    targets. A subclass sets ``_targets`` and defines ``_make_backend``."""

    _targets: List[Dict]

    @abstractmethod
    def _make_backend(self, ab_target: Dict) -> BackendT: ...

    def get_backend(self, name=None, verbose=True, **kwargs) -> BackendT:
        """Return a single backend matching by its name.

        Args:
            name (str): name of the backend
            verbose (bool): if True, will display information about the jobs
                execution.
                Defaults to True.
            **kwargs: additional backend parameters
                (see targets documentation).

        Returns:
            BackendT: backend matching the given name.
        """
        backend = self._get_single_backend(name)
        # We allow to set the options when getting the backend,
        # to align with what we do in the local provider.
        if kwargs:
            backend.update_options(kwargs)
        backend._verbose = verbose
        return backend

    def _get_single_backend(self, name=None, **kwargs) -> BackendT:
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError(
                'More than one backend matches the criteria'
            )
        if not backends:
            raise QiskitBackendNotFoundError('No backend matches the criteria')
        return backends[0]

    def backends(self, name: Optional[str] = None, **kwargs) -> List[BackendT]:
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): if provided, only backends with this name will be
                returned (there should be only one such backend).
            **kwargs: additional parameters for filtering

        Returns:
            List[Backend]: the list of matching backends.
        """
        # backends are loaded from targets dynamically each time to create
        # instances and avoid shared references
        backends_from_targets = [
            self._make_backend(ab_target) for ab_target in self._targets
        ]
        if name:
            backends = [
                backend
                for backend in backends_from_targets
                if backend.name == name
            ]
        else:
            backends = backends_from_targets
        return filter_backends(backends, **kwargs)


class AliceBobRemoteProvider(AliceBobTargetProvider[AliceBobRemoteBackend]):
    """
    Class listing and providing access to all Alice & Bob remote backends.
    """

    def __init__(
        self,
        api_key: str,
        url: str = 'https://api-gcp.alice-bob.com/',
        retries: int = 5,
        wait_between_retries_seconds: int = 1,
    ):
        """
        Args:
            api_key (str): an API key for the Alice & Bob API
            url (str): Base URL of the Alice & Bob API.
                Defaults to 'https://api-gcp.alice-bob.com/'.
        """
        self.client = ApiClient(
            api_key=api_key,
            url=url,
            retries=retries,
            wait_between_retries_seconds=wait_between_retries_seconds,
        )
        self._targets = list_targets(self.client)

        provider_status = get_provider_status()
        if provider_status == ProviderStatus.UNKNOWN:
            logging.warning(
                'Could not determine the latest version of the provider. '
                'Your installation may be outdated.'
            )
        elif provider_status == ProviderStatus.OUTDATED:
            logging.warning(
                'A new version of the provider is available. Install it with '
                '"pip install -U qiskit-alice-bob-provider".'
            )

    def _make_backend(self, ab_target: Dict) -> AliceBobRemoteBackend:
        return AliceBobRemoteBackend(self.client, ab_target)
