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

from qiskit import QuantumCircuit
from qiskit.providers import JobV1
from qiskit.providers.backend import Backend
from qiskit.result import Result
from qiskit_aer.jobs import AerJob


class ProcessorSimulationJob(JobV1):
    """A wrapper for AerJobs containing different versions of the input
    circuits (with and without noise).

    This is mainly useful for debugging purpose."""

    def __init__(
        self,
        backend: Backend,
        wrapped_job: AerJob,
        circuits: List[QuantumCircuit],
        noisy_circuits: List[QuantumCircuit],
        **kwargs,
    ) -> None:
        super().__init__(backend, wrapped_job.job_id(), **kwargs)
        self._circuits = circuits
        self._noisy_circuits = noisy_circuits
        self._wrapped_job = wrapped_job

    def submit(self):
        return self._wrapped_job.submit()

    def result(self) -> Result:
        return self._wrapped_job.result()

    def cancel(self):
        return self._wrapped_job.cancel()

    def status(self):
        return self._wrapped_job.status()

    def circuits(self) -> List[QuantumCircuit]:
        """Return the list of QuantumCircuit augmented with the quantum noise
        instructions."""
        return self._circuits

    def noisy_circuits(self) -> List[QuantumCircuit]:
        """Return the list of QuantumCircuit augmented with the quantum noise
        instructions."""
        return self._noisy_circuits

    @property
    def wrapped_job(self) -> AerJob:
        return self._wrapped_job
