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

from typing import List, Optional

from qiskit.providers import BackendV2, ProviderV1
from qiskit.providers.providerutils import filter_backends

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
        url: str = 'https://api.alice-bob.com/',
        retries: int = 5,
        wait_between_retries_seconds: int = 1,
    ):
        """
        Args:
            api_key (str): an API key for the Alice & Bob API
            url (str): Base URL of the Alice & Bob API.
                Defaults to 'https://api.alice-bob.com/'.
        """
        client = ApiClient(
            api_key=api_key,
            url=url,
            retries=retries,
            wait_between_retries_seconds=wait_between_retries_seconds,
        )
        self._backends = []
        for ab_target in list_targets(client):
            self._backends.append(AliceBobRemoteBackend(client, ab_target))

    def get_backend(self, name=None, **kwargs) -> AliceBobRemoteBackend:
        backend = super().get_backend(name)
        # We allow to set the options when getting the backend,
        # to align with what we do in the local provider.
        if kwargs:
            backend.update_options(kwargs)
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
            List[Backend]: the list of maching backends.
        """
        if name:
            backends = [
                backend for backend in self._backends if backend.name == name
            ]
        else:
            backends = self._backends
        return filter_backends(backends, **kwargs)
