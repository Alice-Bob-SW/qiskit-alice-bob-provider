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

from typing import Dict, Optional

from .client import AliceBobApiException, ApiClient


def create_job(client: ApiClient, target: str, input_params: Dict) -> Dict:
    """Create a job in the Alice & Bob API

    Args:
        client (ApiClient): a client for the Alice & Bob API
        target (str): the name of the target where to execute the program on
        input_params (dict): additional parameters passed to the target

    Returns:
        dict: the API response object, which is the description of the created
            job
    """
    payload = {
        'inputDataFormat': 'HUMAN_QIR',
        'outputDataFormat': 'HISTOGRAM',
        'target': target,
        'inputParams': input_params,
    }
    return client.post('v1/jobs/', json=payload).json()


def get_job(client: ApiClient, job_id: str) -> dict:
    """Get information about a job in the Alice & Bob API

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API

    Returns:
        dict: the API response object, which is the description of the
            requested job
    """
    return client.get(f'v1/jobs/{job_id}').json()


def get_job_metrics(client: ApiClient, job_id: str) -> dict:
    """Get exposed metrics about a job in the Alice & Bob API

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API

    Returns:
        dict: the API response object, containing the recorded metrics.
    """
    return client.get(f'v1/jobs/{job_id}/metrics').json()


def cancel_job(client: ApiClient, job_id: str) -> None:
    """Cancel a job in the Alice & Bob API

    This function will fail with a 409 Conflict if the job cannot be cancelled
    anymore.

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API
    """
    client.delete(f'v1/jobs/{job_id}')


def upload_input(client: ApiClient, job_id: str, input: str) -> None:
    """For a given job, upload the program to execute on the target

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API
        input (str): the program to execute as a string
    """
    client.post(f'v1/jobs/{job_id}/input', files={'input': input})


def _download_file(client: ApiClient, url: str) -> Optional[str]:
    try:
        return client.get(url).content.decode('utf-8')
    except AliceBobApiException as e:
        if e.status_code == 409:
            return None
        raise


def download_input(client: ApiClient, job_id: str) -> Optional[str]:
    """For a given job, download the input program

    This function will return None if no input program was uploaded.

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API

    Returns:
        Optional[str]: the job input program if available
    """
    return _download_file(client, f'v1/jobs/{job_id}/input')


def download_output(client: ApiClient, job_id: str) -> Optional[str]:
    """For a given job, download the output results

    This function will return None if the job is not complete or if the job
    has failed.

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API

    Returns:
        Optional[str]: the job output if available
    """
    return _download_file(client, f'v1/jobs/{job_id}/output')


def download_memory(client: ApiClient, job_id: str) -> Optional[str]:
    """For a given job, download the memory of the program

    This function will return None if the job is not complete or if the job
    has failed.

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API

    Returns:
        Optional[str]: the job memory if available
    """
    return _download_file(client, f'v1/jobs/{job_id}/memory')


def download_transpiled(client: ApiClient, job_id: str) -> Optional[str]:
    """For a given job, download the transpiled program

    This function will return None if the job has not gone through the
    transpilation stage yet, or if the job has failed.

    Args:
        client (ApiClient): a client for the Alice & Bob API
        job_id (str): the ID of the job in the Alice & Bob API

    Returns:
        Optional[str]: the job transpiled program if available
    """
    return _download_file(client, f'v1/jobs/{job_id}/transpiled')
