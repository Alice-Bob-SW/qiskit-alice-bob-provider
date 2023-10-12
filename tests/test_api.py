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

import pytest
import requests
from requests_mock import ANY
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.remote.api import jobs, targets
from qiskit_alice_bob_provider.remote.api.client import ApiClient


def test_authentication(requests_mock: Mocker) -> None:
    base_url = 'https://api.alice-bob.com/'
    api_key = 'foo'
    requests_mock.register_uri(
        ANY,
        ANY,
        json={},
        request_headers={'Authorization': f'Basic {api_key}'},
    )
    client = ApiClient(api_key=api_key, url=base_url)
    jobs.create_job(client, 'TARGET', {})


def test_get_job(requests_mock: Mocker) -> None:
    job_id = 'my-job'
    requests_mock.register_uri('GET', f'/v1/jobs/{job_id}', json={})
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    jobs.get_job(client, job_id)


def test_cancel_job(requests_mock: Mocker) -> None:
    job_id = 'my-job'
    requests_mock.register_uri('DELETE', f'/v1/jobs/{job_id}')
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    jobs.cancel_job(client, job_id)


def test_upload_input(requests_mock: Mocker) -> None:
    job_id = 'my-job'
    content = 'bar'
    requests_mock.register_uri('POST', f'/v1/jobs/{job_id}/input')
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    jobs.upload_input(client, job_id, content)
    payload = requests_mock.request_history[0].text
    assert ' name="input"' in payload
    assert ' filename="input"' in payload
    assert content in payload


def test_download_input(requests_mock: Mocker) -> None:
    job_id = 'my-job'
    content = 'bar'
    requests_mock.register_uri('GET', f'/v1/jobs/{job_id}/input', text=content)
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    response = jobs.download_input(client, job_id)
    assert response == content


def test_download_transpiled(requests_mock: Mocker) -> None:
    job_id = 'my-job'
    content = 'bar'
    requests_mock.register_uri(
        'GET', f'/v1/jobs/{job_id}/transpiled', text=content
    )
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    response = jobs.download_transpiled(client, job_id)
    assert response == content


def test_download_output(requests_mock: Mocker) -> None:
    job_id = 'my-job'
    content = 'bar'
    requests_mock.register_uri(
        'GET', f'/v1/jobs/{job_id}/output', text=content
    )
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    response = jobs.download_output(client, job_id)
    assert response == content


def test_create_job(requests_mock: Mocker) -> None:
    target = 'TARGET'
    input_params = {'foo': 'bar'}
    requests_mock.register_uri('POST', '/v1/jobs/', json={})
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    jobs.create_job(client, target, input_params)
    payload = requests_mock.request_history[0].json()
    assert payload['target'] == target
    assert payload['inputParams'] == input_params


def test_list_targets(requests_mock: Mocker) -> None:
    requests_mock.register_uri('GET', '/v1/targets/', json=[{}, {}])
    client = ApiClient(api_key='foo', url='https://api.alice-bob.com/')
    targets.list_targets(client)


def test_connection_aborted(requests_mock: Mocker) -> None:
    requests_mock.register_uri(
        'GET', '/v1/targets/', exc=requests.ConnectionError
    )
    client = ApiClient(
        api_key='foo',
        url='https://api.alice-bob.com/',
        wait_between_retries_seconds=0,
    )
    with pytest.raises(requests.ConnectionError):
        targets.list_targets(client)
    assert requests_mock.call_count == 5

    requests_mock.register_uri('GET', '/v1/targets/', exc=requests.HTTPError)
    client = ApiClient(
        api_key='foo',
        url='https://api.alice-bob.com/',
        wait_between_retries_seconds=0,
    )
    with pytest.raises(requests.HTTPError):
        targets.list_targets(client)
    assert requests_mock.call_count == 5 + 1

    requests_mock.register_uri(
        'GET', '/v1/targets/', exc=requests.ConnectionError
    )
    client = ApiClient(
        api_key='foo',
        url='https://api.alice-bob.com/',
        retries=4,
        wait_between_retries_seconds=0,
    )
    with pytest.raises(requests.ConnectionError):
        targets.list_targets(client)
    assert requests_mock.call_count == 5 + 1 + 4
