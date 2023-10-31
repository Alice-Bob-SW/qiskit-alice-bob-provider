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

import csv
import logging
from dataclasses import dataclass
from io import StringIO
from typing import Callable, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.providers import JobStatus, JobV1
from qiskit.providers.backend import BackendV2
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.result.models import (
    ExperimentResult,
    ExperimentResultData,
    QobjExperimentHeader,
)

from .api import jobs
from .api.client import ApiClient


@dataclass
class _DownloadedFile:
    """A class caching a file downloaded through the API."""

    # Tells whether this is the last version of the file
    final: bool

    # The content of the file if it was available and downloaded
    content: Optional[str]


class AliceBobRemoteJob(JobV1):
    """A Qiskit job referencing a job executed in the Alice & Bob API"""

    def __init__(
        self,
        backend: BackendV2,
        api_client: ApiClient,
        job_id: str,
        circuit: QuantumCircuit,
    ):
        """A job should not be instantiated manually but created by calling
        :func:`AliceBobRemoteBackend.run` or :func:`qiskit.execute`.

        Args:
            backend (BackendV2): a reference to the backend that created this
                job
            api_client (ApiClient): a client for the Alice & Bob API
            job_id (str): the ID of this job in the Alice & Bob API
            circuit (QuantumCircuit): the Qiskit circuit associated with this
                job
        """
        super().__init__(backend, job_id)
        self._backend_v2 = backend
        self._api_client = api_client
        self._circuit = circuit
        self._last_response: Optional[Dict] = None
        self._status: Optional[JobStatus] = None
        self._counts: Optional[Dict[str, int]] = None
        self._files: Dict[str, _DownloadedFile] = {}

    def _refresh(self) -> None:
        """If the job status is not final, refresh the description of the API
        job stored by this Qiskit job
        """
        if self._status is not None and self._status in JOB_FINAL_STATES:
            return
        self._last_response = jobs.get_job(self._api_client, self.job_id())
        self._status = _ab_event_to_qiskit_status(
            self._last_response['events'][-1]['type']
        )

    def submit(self):
        """Jobs can only be submitted by calling the ``run()`` method of the
        corresponding backend."""
        raise NotImplementedError

    def _get_file(
        self, name: str, func: Callable[[], Optional[str]]
    ) -> Optional[str]:
        """Helper function to download a file from the API and cache it

        If the file is not available but the job status is final, this method
        won't try to call the API if it is called again.
        """
        if name not in self._files:
            self._files[name] = _DownloadedFile(False, None)
        if self._files[name].final:
            return self._files[name].content
        content = func()
        if content is not None or self.status() in JOB_FINAL_STATES:
            self._files[name].final = True
        self._files[name].content = content
        return content

    def _get_input_qir(self) -> Optional[str]:
        return self._get_file(
            'input',
            lambda: jobs.download_input(self._api_client, self.job_id()),
        )

    def _get_transpiled_qir(self) -> Optional[str]:
        return self._get_file(
            'transpiled',
            lambda: jobs.download_transpiled(self._api_client, self.job_id()),
        )

    def _get_output(self) -> Optional[str]:
        return self._get_file(
            'output',
            lambda: jobs.download_output(self._api_client, self.job_id()),
        )

    def _get_counts(self) -> Dict[str, int]:
        """Transform a histogram returned by the API into Qiskit's histogram
        format.
        """
        if self._counts is not None:
            return self._counts
        self._counts = {}
        output = self._get_output()
        assert output is not None
        for row in csv.DictReader(
            StringIO(output),
            fieldnames=['memory', 'count'],
            delimiter=',',
        ):
            self._counts[hex(int(row['memory'], 2))] = int(row['count'])
        return self._counts

    def result(
        self, timeout: Optional[float] = None, wait: float = 5
    ) -> Result:
        """Wait until the job is complete, then return a result."""
        self.wait_for_final_state(timeout=timeout, wait=wait)
        status = self.status()
        assert self._last_response is not None
        success = status == JobStatus.DONE
        return Result(
            job_id=self.job_id(),
            backend_name=self._backend_v2.name,
            backend_version=self._backend_v2.backend_version,
            qobj_id=id(self._circuit),
            success=success,
            status=status.value,
            results=[
                ExperimentResult(
                    shots=self._last_response['inputParams']['nbShots'],
                    success=success,
                    status=str(self._last_response['errors']),
                    header=QobjExperimentHeader(
                        name=self._circuit.name,
                        input_params=self._last_response['inputParams'],
                        memory_slots=self._circuit.num_clbits,
                    ),
                    data=ExperimentResultData(
                        counts=self._get_counts() if success else None,
                        input_qir=self._get_input_qir(),
                        transpiled_qir=self._get_transpiled_qir(),
                    ),
                )
            ],
        )

    def cancel(self) -> None:
        """Attempt to cancel the job.

        This will fail if the job cannot be cancelled anymore."""
        jobs.cancel_job(self._api_client, self.job_id())

    def status(self) -> JobStatus:
        """Return the status of the job, among the values of ``JobStatus``."""
        self._refresh()
        return self._status


# pylint: disable=too-many-return-statements
def _ab_event_to_qiskit_status(event: str) -> JobStatus:
    """_summary_

    Args:
        event (str): an event from the Alice & Bob API, usually the latest
            event about a given job

    Returns:
        JobStatus: a Qiskit job status
    """
    if event == 'CREATED':
        return JobStatus.INITIALIZING
    elif event in {'INPUT_READY', 'COMPILED', 'TRANSPILED'}:
        return JobStatus.QUEUED
    elif event in {'COMPILING', 'TRANSPILING'}:
        return JobStatus.VALIDATING
    elif event == 'EXECUTING':
        return JobStatus.RUNNING
    elif event in {
        'COMPILATION_FAILED',
        'TRANSPILATION_FAILED',
        'EXECUTION_FAILED',
        'TIMED_OUT',
    }:
        return JobStatus.ERROR
    elif event == 'SUCCEEDED':
        return JobStatus.DONE
    elif event == 'CANCELLED':
        return JobStatus.CANCELLED
    logging.warning(
        f'Received unexpected job event {event}. \n'
        'Please ensure you are running the latest version of the Alice & Bob '
        'Provider .'
    )
    # An unknown job status will be considered to be an Error by default.
    return JobStatus.ERROR
