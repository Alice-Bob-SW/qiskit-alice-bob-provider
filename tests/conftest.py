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

from typing import List

import pytest
from requests_mock.mocker import Mocker


def _response(job_id: str, events: List[dict], errors: List[dict]) -> dict:
    return {
        'inputDataFormat': 'HUMAN_QIR',
        'outputDataFormat': 'HISTOGRAM',
        'target': 'SINGLE_CAT_SIMULATOR',
        'inputParams': {'nbShots': 100, 'averageNbPhotons': 4.0},
        'id': job_id,
        'userName': 'john',
        'userId': '42',
        'organizationName': 'acme',
        'events': events,
        'errors': errors,
    }


@pytest.fixture
def successful_job(requests_mock: Mocker) -> Mocker:
    job_id = 'my-job'
    requests_mock.register_uri(
        'POST',
        '/v1/jobs/',
        json=_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-14T14:53:21.772892',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-14T14:56:31.342488',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-14T14:56:33.015329',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-14T14:56:33.029502',
                        },
                    ],
                    [],
                ),
            },
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-14T14:53:21.772892',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-14T14:56:31.342488',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-14T14:56:33.015329',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-14T14:56:33.029502',
                        },
                        {
                            'type': 'TRANSPILING',
                            'createdAt': '2023-05-14T14:56:33.038144',
                        },
                        {
                            'type': 'TRANSPILED',
                            'createdAt': '2023-05-14T14:56:33.171038',
                        },
                        {
                            'type': 'EXECUTING',
                            'createdAt': '2023-05-14T14:56:33.174236',
                        },
                        {
                            'type': 'SUCCEEDED',
                            'createdAt': '2023-05-14T14:56:33.202824',
                        },
                    ],
                    [],
                ),
            },
        ],
    )
    requests_mock.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        text='bar',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/output',
        text='11,12\n10,474\n01,6\n00,508\n',
    )
    return requests_mock


@pytest.fixture
def failed_transpilation_job(requests_mock: Mocker) -> Mocker:
    job_id = 'my-job'
    requests_mock.register_uri(
        'POST',
        '/v1/jobs/',
        json=_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-14T14:53:21.772892',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-14T14:56:31.342488',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-14T14:56:33.015329',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-14T14:56:33.029502',
                        },
                    ],
                    [],
                ),
            },
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-14T14:53:21.772892',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-14T14:56:31.342488',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-14T14:56:33.015329',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-14T14:56:33.029502',
                        },
                        {
                            'type': 'TRANSPILING',
                            'createdAt': '2023-05-14T14:56:33.038144',
                        },
                        {
                            'type': 'TRANSPILATION_FAILED',
                            'createdAt': '2023-05-14T14:56:33.171038',
                        },
                    ],
                    [
                        {
                            'content': {
                                'code': 'OperationNotSupported',
                                'message': (
                                    'Input program requires 3 qubits, '
                                    'the maximum is 1'
                                ),
                            }
                        }
                    ],
                ),
            },
        ],
    )
    requests_mock.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        status_code=409,
        json={
            'error': {
                'code': 409,
                'message': (
                    'Transpilation of job '
                    'f0484867-d154-4c64-b4e4-aa2c4a3fa505 failed'
                ),
            }
        },
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/output',
        status_code=409,
        json={
            'error': {
                'code': 409,
                'message': (
                    'Job f0484867-d154-4c64-b4e4-aa2c4a3fa505 '
                    'failed and has no output'
                ),
            }
        },
    )
    return requests_mock


@pytest.fixture
def failed_execution_job(requests_mock: Mocker) -> Mocker:
    job_id = 'my-job'
    requests_mock.register_uri(
        'POST',
        '/v1/jobs/',
        json=_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-15T20:02:39.148830',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-15T20:03:15.183723',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-15T20:03:15.189539',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-15T20:03:15.192196',
                        },
                    ],
                    [],
                ),
            },
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-15T20:02:39.148830',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-15T20:03:15.183723',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-15T20:03:15.189539',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-15T20:03:15.192196',
                        },
                        {
                            'type': 'TRANSPILING',
                            'createdAt': '2023-05-15T20:03:15.194187',
                        },
                        {
                            'type': 'TRANSPILED',
                            'createdAt': '2023-05-15T20:03:15.338994',
                        },
                        {
                            'type': 'EXECUTING',
                            'createdAt': '2023-05-15T20:03:15.340887',
                        },
                        {
                            'type': 'EXECUTION_FAILED',
                            'createdAt': '2023-05-15T20:03:15.478770',
                        },
                    ],
                    [
                        {
                            'content': {
                                'code': 500,
                                'message': 'An unexpected error happened',
                            }
                        }
                    ],
                ),
            },
        ],
    )
    requests_mock.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        text='bar',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/output',
        status_code=409,
        json={
            'error': {
                'code': 409,
                'message': (
                    'Job f0484867-d154-4c64-b4e4-aa2c4a3fa505 '
                    'failed and has no output'
                ),
            }
        },
    )
    return requests_mock


@pytest.fixture
def cancellable_job(requests_mock: Mocker) -> Mocker:
    job_id = 'my-job'
    requests_mock.register_uri(
        'POST',
        '/v1/jobs/',
        json=_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-15T20:02:39.148830',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-15T20:03:15.183723',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-15T20:03:15.189539',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-15T20:03:15.192196',
                        },
                    ],
                    [],
                ),
            },
            {
                'json': _response(
                    job_id,
                    [
                        {
                            'type': 'CREATED',
                            'createdAt': '2023-05-15T20:02:39.148830',
                        },
                        {
                            'type': 'INPUT_READY',
                            'createdAt': '2023-05-15T20:03:15.183723',
                        },
                        {
                            'type': 'COMPILING',
                            'createdAt': '2023-05-15T20:03:15.189539',
                        },
                        {
                            'type': 'COMPILED',
                            'createdAt': '2023-05-15T20:03:15.192196',
                        },
                        {
                            'type': 'TRANSPILING',
                            'createdAt': '2023-05-15T20:03:15.194187',
                        },
                        {
                            'type': 'TRANSPILED',
                            'createdAt': '2023-05-15T20:03:15.338994',
                        },
                        {
                            'type': 'CANCELLED',
                            'createdAt': '2023-05-15T20:03:15.340887',
                        },
                    ],
                    [],
                ),
            },
        ],
    )
    requests_mock.register_uri(
        'DELETE',
        f'/v1/jobs/{job_id}',
        json=_response(
            job_id,
            [
                {
                    'type': 'CREATED',
                    'createdAt': '2023-05-15T20:02:39.148830',
                },
                {
                    'type': 'INPUT_READY',
                    'createdAt': '2023-05-15T20:03:15.183723',
                },
                {
                    'type': 'COMPILING',
                    'createdAt': '2023-05-15T20:03:15.189539',
                },
                {
                    'type': 'COMPILED',
                    'createdAt': '2023-05-15T20:03:15.192196',
                },
                {
                    'type': 'TRANSPILING',
                    'createdAt': '2023-05-15T20:03:15.194187',
                },
                {
                    'type': 'TRANSPILED',
                    'createdAt': '2023-05-15T20:03:15.338994',
                },
                {
                    'type': 'CANCELLED',
                    'createdAt': '2023-05-15T20:03:15.340887',
                },
            ],
            [],
        ),
    )
    requests_mock.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        text='bar',
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/output',
        status_code=409,
        json={
            'error': {
                'code': 409,
                'message': (
                    'Job f0484867-d154-4c64-b4e4-aa2c4a3fa505 '
                    'was cancelled and has no output'
                ),
            }
        },
    )
    return requests_mock


@pytest.fixture
def failed_validation_job(requests_mock: Mocker) -> Mocker:
    job_id = 'my-job'
    requests_mock.register_uri(
        'POST',
        '/v1/jobs/',
        status_code=400,
        json={
            'error': {
                'code': 400,
                'message': (
                    'Input param \"averageNbPhotons\" '
                    'must be in the range [1, 10].'
                ),
            }
        },
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        status_code=404,
        json={
            'error': {
                'code': 404,
                'message': (
                    'Could not find job '
                    'f0484867-d154-4c64-b4e4-aa2c4a3fa504'
                ),
            }
        },
    )
    requests_mock.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
        status_code=404,
        json={
            'error': {
                'code': 404,
                'message': (
                    'Could not find job '
                    'f0484867-d154-4c64-b4e4-aa2c4a3fa504'
                ),
            }
        },
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        status_code=404,
        json={
            'error': {
                'code': 404,
                'message': (
                    'Could not find job '
                    'f0484867-d154-4c64-b4e4-aa2c4a3fa504'
                ),
            }
        },
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        status_code=404,
        json={
            'error': {
                'code': 404,
                'message': (
                    'Could not find job '
                    'f0484867-d154-4c64-b4e4-aa2c4a3fa504'
                ),
            }
        },
    )
    requests_mock.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/output',
        status_code=404,
        json={
            'error': {
                'code': 404,
                'message': (
                    'Could not find job '
                    'f0484867-d154-4c64-b4e4-aa2c4a3fa504'
                ),
            }
        },
    )
    return requests_mock
