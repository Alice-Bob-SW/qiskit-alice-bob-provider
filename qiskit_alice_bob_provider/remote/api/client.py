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

import urllib.parse

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)


class AliceBobApiException(Exception):
    """Exception raised by the Aice & Bob API client"""

    def __init__(self, response: requests.Response) -> None:
        super().__init__(response.content.decode())
        self.status_code = response.status_code


class ApiClient:
    """Alice & Bob API client

    This wrapper adds authentication and the base URL to all requests.
    """

    def __init__(
        self,
        api_key: str,
        url: str,
        retries: int = 5,
        wait_between_retries_seconds: int = 1,
    ):
        """
        Args:
            api_key (str): an API key provided by Alice & Bob
            url (str): the base URL of the API
        """
        self._url = url
        self._session = requests.Session()
        self._session.headers.update({'Authorization': f'Basic {api_key}'})
        self._retries = retries
        self._wait_between_retries_seconds = wait_between_retries_seconds

    def _request(
        self, method: str, endpoint: str, **kwargs
    ) -> requests.Response:
        url = urllib.parse.urljoin(self._url, endpoint)
        resp = retry(
            reraise=True,
            wait=wait_fixed(self._wait_between_retries_seconds),
            retry=retry_if_exception_type(requests.ConnectionError),
            stop=stop_after_attempt(self._retries),
        )(self._session.request)(method=method, url=url, **kwargs)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise AliceBobApiException(resp) from e
        return resp

    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """GET request to the Alice & Bob API

        Args:
            endpoint (str): Path to the endpoint
            **kwargs: additional parameters passed to the requests library

        Returns:
            requests.Response: the request response
        """
        return self._request(method='get', endpoint=endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """POST request to the Alice & Bob API

        Args:
            endpoint (str): Path to the endpoint
            **kwargs: additional parameters passed to the requests library

        Returns:
            requests.Response: the request response
        """
        return self._request(method='post', endpoint=endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """POST request to the Alice & Bob API

        Args:
            endpoint (str): Path to the endpoint
            **kwargs: additional parameters passed to the requests library

        Returns:
            requests.Response: the request response
        """
        return self._request(method='delete', endpoint=endpoint, **kwargs)
