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

# pylint: disable=redefined-outer-name

from typing import Dict, List

import pytest
from requests_mock.mocker import Mocker


def pytest_addoption(parser):
    parser.addoption(
        '--base-url',
        action='store',
        dest='base_url',
        default=None,
    )
    parser.addoption(
        '--api-key',
        action='store',
        dest='api_key',
        default=None,
    )


@pytest.fixture
def base_url(request):
    """Loads the provided hardware config if any."""
    return request.config.getoption('base_url')


@pytest.fixture
def api_key(request):
    """Loads the provided hardware config if any."""
    return request.config.getoption('api_key')


@pytest.fixture
def single_cat_target() -> dict:
    return {
        'name': 'EMU:1Q:LESCANNE_2020',
        'numQubits': 1,
        'instructions': [
            {'signature': '__quantum__qis__read_result__body:i1 (%Result*)'},
            {'signature': '__quantum__qis__z__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__x__body:void (%Qubit*)'},
            {
                'signature': (
                    '__quantum__qis__mz__body:void ' '(%Qubit*, %Result*)'
                )
            },
            {'signature': '__quantum__qis__m__body:void (%Qubit*, %Result*)'},
            {
                'signature': (
                    '__quantum__qis__measure__body:void (%Qubit*, %Result*)'
                )
            },
            {
                'signature': (
                    '__quantum__qis__mx__body:void (%Qubit*, %Result*)'
                )
            },
            {'signature': '__quantum__qis__reset__body:void (%Qubit*)'},
            {
                'signature': (
                    '__quantum__qis__delay__body:void (double, %Qubit*)'
                )
            },
            {
                'signature': (
                    '__quantum__qis__prepare_x__body:void (i1, %Qubit*)'
                )
            },
            {
                'signature': (
                    '__quantum__qis__prepare_z__body:void (i1, %Qubit*)'
                )
            },
            {'signature': '__quantum__qis__rz__body:void (double, %Qubit*)'},
        ],
        'inputParams': {
            'nbShots': {
                'required': True,
                'default': 1000,
                'constraints': [{'min': 1, 'max': 10000000}],
            },
            'averageNbPhotons': {
                'required': True,
                'default': 4.0,
                'constraints': [{'min': 1.0, 'max': 10.0}],
            },
        },
    }


@pytest.fixture
def all_instructions_target() -> dict:
    return {
        'name': 'ALL_INSTRUCTIONS',
        'numQubits': 7,
        'instructions': [
            {'signature': '__quantum__qis__read_result__body:i1 (%Result*)'},
            {'signature': '__quantum__qis__z__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__x__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__mz__body:void (%Qubit*, %Result*)'},
            {'signature': '__quantum__qis__m__body:void (%Qubit*, %Result*)'},
            {
                'signature': (
                    '__quantum__qis__measure__body:void (%Qubit*, %Result*)'
                )
            },
            {'signature': '__quantum__qis__mx__body:void (%Qubit*, %Result*)'},
            {'signature': '__quantum__qis__reset__body:void (%Qubit*)'},
            {
                'signature': (
                    '__quantum__qis__delay__body:void (double, %Qubit*)'
                )
            },
            {
                'signature': (
                    '__quantum__qis__prepare_x__body:void (i1, %Qubit*)'
                )
            },
            {
                'signature': (
                    '__quantum__qis__prepare_z__body:void (i1, %Qubit*)'
                )
            },
            {'signature': '__quantum__qis__rx__body:void (double, %Qubit*)'},
            {'signature': '__quantum__qis__ry__body:void (double, %Qubit*)'},
            {'signature': '__quantum__qis__rz__body:void (double, %Qubit*)'},
            {
                'signature': (
                    '__quantum__qis__cnot__body:void (%Qubit*, %Qubit*)'
                )
            },
            {'signature': '__quantum__qis__cx__body:void (%Qubit*, %Qubit*)'},
            {
                'signature': (
                    '__quantum__qis__ccx__body:void '
                    '(%Qubit*, %Qubit*, %Qubit*)'
                )
            },
            {
                'signature': (
                    '__quantum__qis__toffoli__body:void '
                    '(%Qubit*, %Qubit*, %Qubit*)'
                )
            },
            {'signature': '__quantum__qis__cz__body:void (%Qubit*, %Qubit*)'},
            {'signature': '__quantum__qis__y__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__h__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__s__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__t__body:void (%Qubit*)'},
            {
                'signature': (
                    '__quantum__qis__rzz__body:void (double, %Qubit*, %Qubit*)'
                )
            },
            {'signature': '__quantum__qis__barrier__body:void ()'},
            {
                'signature': (
                    '__quantum__qis__swap__body:void (%Qubit*, %Qubit*)'
                )
            },
        ],
        'inputParams': {
            'nbShots': {
                'required': True,
                'default': 1000,
                'constraints': [{'min': 1, 'max': 10000000}],
            },
            'averageNbPhotons': {
                'required': False,
                'default': None,
                'constraints': [],
            },
        },
    }


@pytest.fixture
def h_target() -> dict:
    return {
        'name': 'H',
        'numQubits': 1,
        'instructions': [
            {'signature': '__quantum__qis__h__body:void (%Qubit*)'}
        ],
        'inputParams': {
            'nbShots': {
                'required': True,
                'default': 1000,
                'constraints': [{'min': 1, 'max': 10000000}],
            },
            'averageNbPhotons': {
                'required': False,
                'default': None,
                'constraints': [],
            },
        },
    }


@pytest.fixture
def h_t_target() -> dict:
    return {
        'name': 'H_T',
        'numQubits': 1,
        'instructions': [
            {'signature': '__quantum__qis__h__body:void (%Qubit*)'},
            {'signature': '__quantum__qis__t__body:void (%Qubit*)'},
        ],
        'inputParams': {
            'nbShots': {
                'required': True,
                'default': 1000,
                'constraints': [{'min': 1, 'max': 10000000}],
            },
            'averageNbPhotons': {
                'required': False,
                'default': None,
                'constraints': [],
            },
        },
    }


@pytest.fixture
def targets(
    single_cat_target: Dict,
    all_instructions_target: Dict,
    h_target: Dict,
    h_t_target: Dict,
) -> List[Dict]:
    return [single_cat_target, all_instructions_target, h_target, h_t_target]


@pytest.fixture
def mocked_targets(targets: List[Dict], requests_mock: Mocker) -> Mocker:
    requests_mock.register_uri(
        'GET',
        '/v1/targets/',
        json=targets,
    )
    return requests_mock


def _job_response(job_id: str, events: List[Dict], errors: List[Dict]) -> dict:
    return {
        'inputDataFormat': 'HUMAN_QIR',
        'outputDataFormat': 'HISTOGRAM',
        'target': 'EMU:1Q:LESCANNE_2020',
        'inputParams': {'nbShots': 100, 'averageNbPhotons': 4.0},
        'id': job_id,
        'userName': 'john',
        'userId': '42',
        'organizationName': 'acme',
        'events': events,
        'errors': errors,
    }


@pytest.fixture
def successful_job(mocked_targets: Mocker) -> Mocker:
    job_id = 'my-job'
    mocked_targets.register_uri(
        'POST',
        '/v1/jobs/',
        json=_job_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _job_response(
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
                'json': _job_response(
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
    mocked_targets.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        text='bar',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/output',
        text='11,12\n10,474\n01,6\n00,508\n',
    )
    return mocked_targets


@pytest.fixture
def failed_transpilation_job(mocked_targets: Mocker) -> Mocker:
    job_id = 'my-job'
    mocked_targets.register_uri(
        'POST',
        '/v1/jobs/',
        json=_job_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _job_response(
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
                'json': _job_response(
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
    mocked_targets.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    mocked_targets.register_uri(
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
    mocked_targets.register_uri(
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
    return mocked_targets


@pytest.fixture
def failed_execution_job(mocked_targets: Mocker) -> Mocker:
    job_id = 'my-job'
    mocked_targets.register_uri(
        'POST',
        '/v1/jobs/',
        json=_job_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _job_response(
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
                'json': _job_response(
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
    mocked_targets.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        text='bar',
    )
    mocked_targets.register_uri(
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
    return mocked_targets


@pytest.fixture
def cancellable_job(mocked_targets: Mocker) -> Mocker:
    job_id = 'my-job'
    mocked_targets.register_uri(
        'POST',
        '/v1/jobs/',
        json=_job_response(
            job_id,
            [{'type': 'CREATED', 'createdAt': '2023-05-14T14:53:21.772892'}],
            [],
        ),
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}',
        [
            {
                'json': _job_response(
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
                'json': _job_response(
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
    mocked_targets.register_uri(
        'DELETE',
        f'/v1/jobs/{job_id}',
        json=_job_response(
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
    mocked_targets.register_uri(
        'POST',
        f'/v1/jobs/{job_id}/input',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/input',
        text='foo',
    )
    mocked_targets.register_uri(
        'GET',
        f'/v1/jobs/{job_id}/transpiled',
        text='bar',
    )
    mocked_targets.register_uri(
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
    return mocked_targets


@pytest.fixture
def failed_validation_job(mocked_targets: Mocker) -> Mocker:
    job_id = 'my-job'
    mocked_targets.register_uri(
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
    mocked_targets.register_uri(
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
    mocked_targets.register_uri(
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
    mocked_targets.register_uri(
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
    mocked_targets.register_uri(
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
    mocked_targets.register_uri(
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
    return mocked_targets
