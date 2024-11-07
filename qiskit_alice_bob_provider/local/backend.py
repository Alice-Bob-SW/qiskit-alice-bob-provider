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

from typing import List, Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import PassManager, Target
from qiskit_aer import AerSimulator
from qiskit_aer.backends.aerbackend import AerBackend

from ..processor.description import ProcessorDescription
from ..processor.utils import get_init_params
from .job import ProcessorSimulationJob
from .quantum_errors import build_quantum_error_passes
from .readout_errors import build_readout_noise_model
from .target import processor_to_target


class ProcessorSimulator(BackendV2):
    """A Qiskit backend enabling transpilation to and simulation of an
    arbitrary quantum processor (QPU), as described by a ProcessorDescription
    instance.

    Although the ProcessorDescription class provides a richer representation
    of a QPU than Qiskit's Target and NoiseModel can, the ProcessorSimulator
    class should feel like any other Qiskit backend.

    ```
    from qiskit import QuantumCircuit
    from qiskit_alice_bob_provider.local import ProcessorSimulator
    from qiskit_alice_bob_provider.physical_cat import PhysicalCatProcessor


    circ = QuantumCircuit(2, 2)
    # Building the circuit....

    processor = PhysicalCatProcessor()
    backend = ProcessorSimulator(processor)

    # Transpile and schedule the circuit
    transpiled = transpile(circ, backend)

    # Simulate the circuit
    backend.run(transpiled)
    ```
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        processor: ProcessorDescription,
        execution_backend: AerBackend = AerSimulator(),
        name: Optional[str] = None,
        scheduling_stage_plugin: str = 'ab_asap',
        translation_stage_plugin: str = 'state_preparation',
    ):
        """A Qiskit backend enabling transpilation to and simulation of an
        arbitrary quantum processor, as described by a ProcessorDescription
        instance.

        Args:
            processor (ProcessorDescription): the description of the quantum
                processor to simulate
            execution_backend (AerBackend, optional): the Qiskit simulator used
                in the background for simulation. Defaults to AerSimulator().
            name (Optional[str], optional): an optional name for the backend.
        """
        super().__init__(name=name, backend_version=1)
        self._processor = processor
        self._target = processor_to_target(processor)
        self._execution_backend = execution_backend
        self._execution_backend.set_option('n_qubits', self._target.num_qubits)
        self._noise_pass_manager = PassManager(
            build_quantum_error_passes(processor)
        )
        self._noise_model = build_readout_noise_model(processor)
        self._scheduling_stage_plugin = scheduling_stage_plugin
        self._translation_stage_plugin = translation_stage_plugin

    @property
    def target(self) -> Target:
        """The Qiskit Target representing the QPU associated to this backend
        (ProcessorSimulator) for the Qiskit transpiler.

        Warning: the Qiskit Target does not accurately describe a QPU like
        a ProcessorDescription does. Due to its implementation, it suffers
        multiple limitations and should not be trusted.
        In this backend, we only use it as a vessel to provide the Qiskit
        transpiler with a custom implementation of InstructionDurations.
        To learn more about the limitations of Qiskit Targets, please read
        the docstring  of :func:`processor_to_target`."""
        return self._target

    @classmethod
    def _default_options(cls) -> Options:
        return AerSimulator().options

    @property
    def options(self):
        """Return the options of the underlying execution backend.

        The execution backend is AerSimulator unless otherwise specified.
        """
        return self._execution_backend.options

    def set_options(self, **fields):
        """Set the options fields of the underlying execution backend.

        The execution backend is AerSimulator unless otherwise specified.
        """
        self._execution_backend.set_options(**fields)

    def __repr__(self) -> str:
        return f'<ProcessorSimulator(name={self.name})>'

    @property
    def max_circuits(self) -> Optional[int]:
        return None

    def run(
        self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options
    ) -> ProcessorSimulationJob:
        """Simulate the execution of one or multiple circuits on a QPU.

        Note that these circuits should be already transpiled and scheduled,
        otherwise the simulation results will be meaningless.
        To do so, apply function :func:`transpile` before calling
        `ProcessorSimulator.run`.

        Args:
            run_input (Union[QuantumCircuit, List[QuantumCircuit]]): one or
                multiple circuits to simulate.
            **options: additional arguments are interpreted as options for
                the underlying execution backend, usually an instance of
                AerSimulator.

        Returns:
            ProcessorSimulationJob: A wrapper of a submitted asynchronous
                AerJob

        Raises:
            ValueError: if a custom A&B option is passed to backend.run()
        """
        if options:
            # Check that the options passed are not custom A&B options of the
            # processor because they will be ignored in this function.
            processor_params = get_init_params(self._processor.__class__)
            for option in options:
                if option not in processor_params:
                    continue
                raise ValueError(
                    f'The Alice & Bob custom option {option} is not allowed '
                    'for backend.run() with local provider. '
                    'You should pass it to get_backend() instead.'
                )

        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        else:
            circuits = run_input

        # This step inserts all the quantum errors Kraus maps into the
        # circuits.
        noisy_circuits = self._noise_pass_manager.run(circuits)

        # Instructions that cannot be simulated remain in the circuit after
        # the previous step. For example: Initialize(+), MeasureX().
        # Decomposing the circuits ensures those instructions are decomposed
        # into smaller instructions that the simulator can work with.
        # We then need to decompose once again the 'state_preparation' gate
        # introduced when decomposing the 'initialize' gate. The second call to
        # decompose() will decompose them into their clifford representation.
        # Note: this will cause instructions that may not be in the processor
        # instruction set to appear. This is not an issue though, because at
        # this stage the noise has already been inserted and all
        # transformations done by the Qiskit transpiler don't change the
        # simulation results.
        decomposed = [
            c.decompose().decompose(gates_to_decompose=['state_preparation'])
            for c in noisy_circuits
        ]

        job = self._execution_backend.run(
            decomposed, noise_model=self._noise_model, **options
        )

        return ProcessorSimulationJob(
            backend=self,
            wrapped_job=job,
            circuits=circuits,
            noisy_circuits=noisy_circuits,
        )

    def get_scheduling_stage_plugin(self) -> str:
        """This hook tells Qiskit to schedule the circuit with the specified
        scheduling plugin (e.g., with the 'ab_asap' method)"""
        return self._scheduling_stage_plugin

    def get_translation_stage_plugin(self) -> str:
        """This hook tells Qiskit to run the transpilation passes using the
        specified translation plugin
        (e.g. translation_plugin.LocalStatePreparationPlugin)"""
        return self._translation_stage_plugin
