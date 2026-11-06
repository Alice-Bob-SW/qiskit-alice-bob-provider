import base64
import json
from typing import Dict

from qat.comm.exceptions.ttypes import QPUException
from qat.lang import Program
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

from ..remote.backend import AliceBobTargetBackend
from ..remote.job import build_job_result


class AliceBobQaptivaJob(JobV1):
    """A Qiskit job run through a myQLM server. The run waits for the
    counts, so the job is done when it is created."""

    def __init__(self, backend: 'AliceBobQaptivaBackend', result: Result):
        super().__init__(backend, result.job_id)
        self._result = result

    def submit(self):
        """Jobs can only be submitted by calling the ``run()`` method of the
        corresponding backend."""
        raise NotImplementedError

    def result(self) -> Result:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE


class AliceBobQaptivaBackend(AliceBobTargetBackend):
    """Class representing the targets of the Alice & Bob API, reached
    through an anb-myqlm-server."""

    def __init__(self, remote_qpu, target_description: Dict):
        """This class should be instantiated by the AliceBobQaptivaProvider.

        Args:
            remote_qpu: a myQLM QPU that reaches the anb-myqlm-server.
            target_description (dict): a description of the API target behind
                the Qiskit backend, as returned by the Alice & Bob API targets
                endpoint.
        """
        super().__init__(target_description)
        self._remote_qpu = remote_qpu

    def __repr__(self) -> str:
        return f'<AliceBobQaptivaBackend(name={self.name})>'

    def run(self, run_input: QuantumCircuit, **kwargs) -> AliceBobQaptivaJob:
        """Run the quantum circuit through the myQLM server, and wait for the
        counts.

        Args:
            run_input (QuantumCircuit): a Qiskit circuit
            **kwargs: additional arguments are interpreted as backend options.
                List all options by calling
                :func:`AliceBobQaptivaBackend.options`

        Returns:
            AliceBobQaptivaJob: A Qiskit job that holds the result.
        """
        run_input, qir, input_params = self._prepare_run(run_input, kwargs)
        if input_params.get('memory', False):
            raise ValueError(
                'The memory option is not supported through Qaptiva: the '
                'server returns only the counts.'
            )
        program = Program()
        program.qalloc(run_input.num_qubits)
        job = program.to_circ().to_job(nbshots=input_params['nbShots'])
        job.meta_data = {
            'qir': base64.b64encode(qir.encode()).decode(),
            'backend': self.name,
            'input_params': json.dumps(input_params),
        }
        try:
            meta_data = self._remote_qpu.submit(job).meta_data
        except QPUException as exc:
            raise QiskitError(
                f'The myQLM server failed: {exc.message}'
            ) from exc
        counts = json.loads(meta_data['counts'])
        return AliceBobQaptivaJob(
            self,
            build_job_result(
                backend=self,
                job_id=meta_data['job_id'],
                circuit=run_input,
                status=JobStatus.DONE,
                input_params=input_params,
                errors=[],
                counts={hex(int(bits, 2)): n for bits, n in counts.items()},
                memory=None,
                input_qir=qir,
                transpiled_qir=None,
            ),
        )
