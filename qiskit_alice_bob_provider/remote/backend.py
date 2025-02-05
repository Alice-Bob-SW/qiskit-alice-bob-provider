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
from copy import deepcopy
from typing import Any, Dict

import numpy as np
from pydantic.alias_generators import to_camel, to_snake
from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import PassManager, Target
from qiskit_qir import to_qir_module

from ..plugins.state_preparation import EnsurePreparationPass
from .api import jobs
from .api.client import ApiClient
from .display import display_current_line, display_new_line
from .job import AliceBobRemoteJob
from .qir_to_qiskit import ab_target_to_qiskit_target


class AliceBobRemoteBackend(BackendV2):
    """Class representing the targets accessible via the Alice & Bob API."""

    def __init__(self, api_client: ApiClient, target_description: Dict):
        """This class should be instantiated by the AliceBobRemoteProvider.

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
        self._translation_plugin = _determine_translation_plugin(self._target)
        self._verbose = True

    def __repr__(self) -> str:
        return f'<AliceBobRemoteBackend(name={self.name})>'

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
        in translation_plugin.StatePreparationPlugin.
        This function is called before we start the circuit translation,
        and we therefore also use it to inform that we are starting this step.
        """
        return self._translation_plugin

    def update_options(self, option_updates: Dict[str, Any]) -> Options:
        update_options_object(self.options, option_updates)
        return self.options

    def run(self, run_input: QuantumCircuit, **kwargs) -> AliceBobRemoteJob:
        """Run the quantum circuit on the Alice & Bob backend by calling the
        Alice & Bob API.

        Args:
            run_input (QuantumCircuit): a Qiskit circuit
            **kwargs: additional arguments are interpreted as backend options.
                List all options by calling
                :func:`AliceBobRemoteBackend.options`

        Returns:
            AliceBobRemoteJob: A Qiskit job that is a reference to the job
                created in the Alice & Bob API. The job is already started.
                Wait for the results by calling
                :func:`AliceBobRemoteJob.result`.
        """
        if not isinstance(run_input, QuantumCircuit):
            # Qiskit's execute() allows also for list[QuantumCircuit],
            # Schedule and list[Schedule] as inputs. These types of experiments
            # are not yet handled by the provider.
            raise NotImplementedError(
                'Experiments input not supported by the Alice & Bob provider. '
                'Please provide an instance of QuantumCircuit.'
            )
        if self._verbose:
            display_new_line()
            display_current_line('Sending circuit to the API...')
        # Copy value instead of reference to avoid modifying backend options
        # definitively
        new_options = deepcopy(self.options)
        update_options_object(new_options, kwargs)
        input_params = _ab_input_params_from_options(new_options)
        job = jobs.create_job(self._api_client, self.name, input_params)
        run_input = PassManager([EnsurePreparationPass()]).run(run_input)
        jobs.upload_input(
            self._api_client, job['id'], _qiskit_to_qir(run_input)
        )
        return AliceBobRemoteJob(
            backend=self,
            api_client=self._api_client,
            job_id=job['id'],
            circuit=run_input,
            verbose=self._verbose,
            has_memory=input_params.get('memory', False),
        )


def _qiskit_to_qir(circuit: QuantumCircuit) -> str:
    """Transform a Qiskit circuit into a human-readable QIR program"""
    return str(to_qir_module(circuit)[0])


def _options_from_ab_target(ab_target: Dict) -> Options:
    """Extract Qiskit options from an Alice & Bob target description"""
    options = Options()
    for camel_name, desc in ab_target['inputParams'].items():
        name = to_snake(camel_name)
        if name == 'nb_shots':  # special case
            name = 'shots'
        options.update_options(**{name: desc['default']})
        for constraint in desc['constraints']:
            if 'min' in constraint and 'max' in constraint:
                options.set_validator(
                    name, (constraint['min'], constraint['max'])
                )
    return options


def _ab_input_params_from_options(options: Options) -> Dict:
    """Extract Qiskit options from an Alice & Bob target description"""
    input_params: Dict[str, Any] = {}
    for snake_name, value in options.__dict__.items():
        name = to_camel(snake_name)
        if name == 'shots':  # special case
            name = 'nbShots'
        if isinstance(value, np.generic):
            value = value.item()
        input_params[name] = value
    return input_params


def _determine_translation_plugin(target: Target) -> str:
    """Choose which translation plugin to apply, depending on the gate set of
    the backend."""
    instructions = {instr.name for instr, _ in target.instructions}
    if any(
        rotation in instructions for rotation in ['u', 'rx', 'ry', 'rz']
    ) or any(gate not in instructions for gate in ['h', 't']):
        return 'state_preparation'
    return 'sk_synthesis'


def update_options_object(options: Options, option_updates: Dict[str, Any]):
    for key, value in option_updates.items():
        if not hasattr(options, key):
            raise ValueError(f'Backend does not support option "{key}"')
        options.update_options(**{key: value})
