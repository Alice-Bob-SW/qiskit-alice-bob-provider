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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Delay, Measure, Reset
from qiskit.circuit.library import RZGate, XGate, ZGate
from qiskit.circuit.parameter import Parameter
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit_qir import to_qir_module

from .api import jobs
from .api.client import ApiClient
from .custom_instructions import MeasureX
from .job import AliceBobJob


class CatSimulatorBackend(BackendV2):
    """Class representing the single cat simulator target accessible via the
    Alice & Bob API"""

    def __init__(self, api_client: ApiClient):
        """This class should be instantiated by the AliceBobProvider.

        Args:
            api_client (ApiClient): a client for the Alice & Bob API.
        """
        super().__init__(name='SINGLE_CAT_SIMULATOR', backend_version=1)
        self._api_client = api_client

    @property
    def target(self) -> Target:
        target = Target(num_qubits=1)
        phi = Parameter('ϕ')
        duration = Parameter('t')
        target.add_instruction(Delay(duration, unit='us'))
        target.add_instruction(Reset(), name='prepare_0')
        target.add_instruction(Initialize([0, 1]), name='prepare_1')
        target.add_instruction(
            Initialize([1 / np.sqrt(2), 1 / np.sqrt(2)]), name='prepare_plus'
        )
        target.add_instruction(
            Initialize([1 / np.sqrt(2), -1 / np.sqrt(2)]), name='prepare_minus'
        )
        target.add_instruction(MeasureX())
        target.add_instruction(Measure())
        target.add_instruction(XGate())
        target.add_instruction(ZGate())
        target.add_instruction(RZGate(phi))
        return target

    @classmethod
    def _default_options(cls) -> Options:
        default = Options(shots=1000, average_nb_photons=3)
        default.set_validator('shots', (1, 10000000))
        default.set_validator('average_nb_photons', (1, 10))
        return default

    @property
    def max_circuits(self):
        return 1

    def get_translation_stage_plugin(self):
        """This hook tells Qiskit to run the transpilation passes contained
        in translation_plugin.StatePreparationPlugin"""
        return 'state_preparation'

    def run(self, run_input: QuantumCircuit, **kwargs) -> AliceBobJob:
        """Run the quantum circuit on the Alice & Bob backend by calling the
        Alice & Bob API.

        Args:
            run_input (QuantumCircuit): a Qiskit circuit
            **kwargs: additional arguments are interpreted as backend options.
                List all options by calling :func:`CatSimulatorBackend.options`

        Returns:
            AliceBobJob: A Qiskit job that is a reference to the job created in
                the Alice & Bob API. The job is already started. Wait for the
                results by calling :func:`AliceBobJob.result`.
        """
        options = self.options
        for key, value in kwargs.items():
            if not hasattr(options, key):
                raise ValueError(f'Backend does not support option "{key}"')
            options[key] = value
        input_params = {
            'nbShots': options['shots'],
            'averageNbPhotons': options['average_nb_photons'],
        }
        job = jobs.create_job(self._api_client, self.name, input_params)
        jobs.upload_input(
            self._api_client, job['id'], _qiskit_to_qir(run_input)
        )
        return AliceBobJob(
            backend=self,
            api_client=self._api_client,
            job_id=job['id'],
            circuit=run_input,
        )


def _qiskit_to_qir(circuit: QuantumCircuit) -> str:
    """Transform a Qiskit circuit into a human-readable QIR program"""
    return str(to_qir_module(circuit)[0])
