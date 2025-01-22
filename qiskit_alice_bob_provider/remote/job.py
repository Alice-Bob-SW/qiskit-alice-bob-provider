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
import time
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, Dict, Optional

from qiskit import QuantumCircuit
from qiskit.providers import JobStatus, JobV1
from qiskit.providers.backend import BackendV2
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.result import Result
from qiskit.result.models import (
    ExperimentResult,
    ExperimentResultData,
    QobjExperimentHeader,
)

from .api import jobs
from .api.client import ApiClient
from .api.models import AliceBobEventType
from .display import display_current_line


@dataclass
class _DownloadedFile:
    """A class caching a file downloaded through the API."""

    # Tells whether this is the last version of the file
    final: bool

    # The content of the file if it was available and downloaded
    content: Optional[str]


# pylint: disable=too-many-instance-attributes
class AliceBobRemoteJob(JobV1):
    """A Qiskit job referencing a job executed in the Alice & Bob API"""

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        backend: BackendV2,
        api_client: ApiClient,
        job_id: str,
        circuit: QuantumCircuit,
        verbose: bool,
        has_memory: bool = False,
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
        self._verbose = verbose
        self._last_response: Optional[Dict] = None
        self._ab_status: Optional[AliceBobEventType] = None
        self._status: Optional[JobStatus] = None
        self._counts: Optional[Dict[str, int]] = None
        self._memory: Optional[list[str]] = None
        self._has_memory = has_memory
        self._files: Dict[str, _DownloadedFile] = {}
        self._metrics: Dict[str, Any] = {}

    def _refresh(self) -> None:
        """If the job status is not final, refresh the description of the API
        job stored by this Qiskit job
        """
        if self._status is not None and self._status in JOB_FINAL_STATES:
            return
        self._last_response = jobs.get_job(self._api_client, self.job_id())
        self._ab_status = AliceBobEventType(
            self._last_response['events'][-1]['type']
        )
        self._status = self._ab_status.to_qiskit_status()

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

    def _get_memory_output(self) -> Optional[str]:
        return self._get_file(
            'memory',
            lambda: jobs.download_memory(self._api_client, self.job_id()),
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

    def _get_memory(self) -> Optional[list[str]]:
        if self._memory is not None:
            return self._memory

        output = self._get_memory_output()

        if output is not None:
            self._memory = output.split(',')

        return self._memory

    def _get_metrics(self) -> Dict[str, Any]:
        self._metrics = jobs.get_job_metrics(self._api_client, self.job_id())
        return self._metrics

    def _monitor_state(
        self, timeout: Optional[float] = None, wait: float = 2
    ) -> None:
        start_time = time.time()
        status = self.status()

        while status not in JOB_FINAL_STATES:
            if (
                self._verbose
                and self._ab_status == AliceBobEventType.INPUT_READY
            ):
                display_current_line(
                    f'Job {self.job_id()} is waiting to be compiled.'
                )
            if self._verbose and self._ab_status in {
                AliceBobEventType.COMPILING,
                AliceBobEventType.TRANSPILING,
            }:
                # We take the shortcut of assuming compilation + transpilation
                # are the same things at the moment.
                display_current_line(f'Job {self.job_id()} is being compiled.')
            if (
                self._verbose
                and self._ab_status == AliceBobEventType.TRANSPILED
            ):
                display_current_line(
                    f'Job {self.job_id()} is waiting to be executed.'
                )
            if (
                self._verbose
                and self._ab_status == AliceBobEventType.EXECUTING
            ):
                display_current_line(f'Job {self.job_id()} is being executed.')

            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(
                    f'Timeout while waiting for job {self.job_id()}.'
                )

            time.sleep(wait)
            status = self.status()

        if self._verbose and self._ab_status == AliceBobEventType.SUCCEEDED:
            metrics = self._get_metrics()
            success_msg = f'Job {self.job_id()} finished successfully.'
            if 'qpu_duration_ns' in metrics and isinstance(
                metrics['qpu_duration_ns'], int
            ):
                success_msg += (
                    ' Time spent executing on QPU: '
                    f'{_format_nanoseconds(metrics["qpu_duration_ns"])}.'
                )
            elif 'simulation_duration_ns' in metrics and isinstance(
                metrics['simulation_duration_ns'], int
            ):
                success_msg += (
                    ' Time spent simulating: '
                    f'{_format_nanoseconds(metrics["simulation_duration_ns"])}'
                    '.'
                )
            display_current_line(success_msg)
        # For the other final states, Qiskit is already displaying the error
        # with the relevant details.

    def result(
        self, timeout: Optional[float] = None, wait: float = 2
    ) -> Result:
        """Wait until the job is complete, then return a result."""
        self._monitor_state(timeout, wait)
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
                        memory=self._get_memory()
                        if success and self._has_memory
                        else None,
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


def _format_nanoseconds(time_ns: int) -> str:
    """Format a given number of nanoseconds to a string containing a unit
    and the time value converted to this unit in the most readable way.
    This will convert the nanoseconds either to microseconds, milliseconds,
    seconds or minutes."""
    if time_ns < 1e3:  # < 1us
        return f'{time_ns}ns'
    elif time_ns < 1e6:  # < 1ms
        return f'{(time_ns / 1e3):.2f}us'
    elif time_ns < 1e9:  # < 1s
        return f'{(time_ns / 1e6):.2f}ms'
    elif time_ns < 60e9:  # < 1min
        return f'{(time_ns / 1e9):.2f}s'
    else:  # min & seconds format
        minutes, seconds = divmod(time_ns / 1e9, 60)
        return f'{minutes}min {seconds:.0f}s'
