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
from typing import List, Optional

from qiskit.providers import BackendV2, ProviderV1
from qiskit.providers.providerutils import filter_backends

from qiskit_alice_bob_provider.remote.api.version import (
    ProviderStatus,
    get_provider_status,
)

from .api.client import ApiClient
from .api.targets import list_targets
from .backend import AliceBobRemoteBackend


class AliceBobRemoteProvider(ProviderV1):
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

    def get_backend(
        self, name=None, verbose=True, **kwargs
    ) -> AliceBobRemoteBackend:
        """Return a single backend matching by its name.

        Args:
            name (str): name of the backend
            verbose (bool): if True, will display information about the jobs
                execution.
                Defaults to True.
            **kwargs: additional backend parameters
                (see targets documentation).

        Returns:
            AliceBobRemoteBackend: backend matching the given name.
        """
        backend = super().get_backend(name)
        # We allow to set the options when getting the backend,
        # to align with what we do in the local provider.
        if kwargs:
            backend.update_options(kwargs)

        # pylint: disable=protected-access
        backend._verbose = verbose
        return backend

    def backends(
        self, name: Optional[str] = None, **kwargs
    ) -> List[BackendV2]:
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
            AliceBobRemoteBackend(self.client, ab_target)
            for ab_target in self._targets
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
