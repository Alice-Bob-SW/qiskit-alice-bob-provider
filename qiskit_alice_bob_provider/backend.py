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

from typing import Dict

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit_qir import to_qir_module

from .api import jobs
from .api.client import ApiClient
from .job import AliceBobJob
from .qir_to_qiskit import ab_target_to_qiskit_target
from .utils import camel_to_snake_case


class AliceBobBackend(BackendV2):
    """Class representing the single cat simulator target accessible via the
    Alice & Bob API"""

    def __init__(self, api_client: ApiClient, target_description: Dict):
        """This class should be instantiated by the AliceBobProvider.

        Args:
            api_client (ApiClient): a client for the Alice & Bob API.
            target_description (dict): a description of the API target behind
                the Qiskit backend, as returned by the Alice & Bob API targets
                endpoint.
        """
        super().__init__(name=target_description['name'], backend_version=1)
        self._api_client = api_client
        self._target = ab_target_to_qiskit_target(target_description)
        self._options = _options_from_ab_target(target_description)

    def __repr__(self) -> str:
        return f'<AliceBobBackend(name={self.name})>'

    @property
    def target(self) -> Target:
        return self._target

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

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
                List all options by calling :func:`AliceBobBackend.options`

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


def _options_from_ab_target(ab_target: Dict) -> Options:
    """Extract Qiskit options from an Alice & Bob target description"""
    options = Options()
    for camel_name, desc in ab_target['inputParams'].items():
        name = camel_to_snake_case(camel_name)
        if name == 'nb_shots':  # special case
            name = 'shots'
        options[name] = desc['default']
        for constraint in desc['constraints']:
            if 'min' in constraint and 'max' in constraint:
                options.set_validator(
                    name, (constraint['min'], constraint['max'])
                )
    return options
