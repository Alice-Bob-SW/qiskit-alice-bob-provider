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

from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    overload,
)

import numpy as np
from qiskit.providers import BackendV2, ProviderV1

from ..processor.custom_cat import CustomCat
from ..processor.interpolated_cat import InterpolatedCatProcessor
from ..processor.logical_cat import LogicalCatProcessor
from ..processor.physical_cat import PhysicalCatProcessor
from ..processor.serialization.model import SerializedProcessor
from .backend import ProcessorSimulator
from .coupling_maps import circular_map, rectangular_map
from .noise_model import NoiseFunction, TimeFunction

_PARENT_DIR = Path(__file__).parent


class AliceBobLocalProvider(ProviderV1):
    """Class listing a number of preset backends simulating cat qubit based
    quantum processors.

    Simulations will run locally on the machine where the backend is invoked.
    """

    def __init__(self) -> None:
        self._backend_builders: Dict[
            str, Callable[..., ProcessorSimulator]
        ] = {}
        self._backend_builders['EMU:6Q:PHYSICAL_CATS'] = partial(
            self.build_physical_backend,
            n_qubits=6,
            coupling_map=circular_map(6),
            name='EMU:6Q:PHYSICAL_CATS',
        )
        self._backend_builders['EMU:40Q:PHYSICAL_CATS'] = partial(
            self.build_physical_backend,
            n_qubits=40,
            coupling_map=rectangular_map(5, 8),
            name='EMU:40Q:PHYSICAL_CATS',
        )
        self._backend_builders['EMU:40Q:LOGICAL_TARGET'] = partial(
            self.build_logical_backend,
            n_qubits=40,
            distance=15,
            kappa_1=100,
            kappa_2=10_000_000,
            average_nb_photons=19,
            name='EMU:40Q:LOGICAL_TARGET',
        )
        self._backend_builders['EMU:40Q:LOGICAL_NOISELESS'] = partial(
            self.build_logical_noiseless_backend,
            n_qubits=40,
            name='EMU:40Q:LOGICAL_NOISELESS',
        )
        self._backend_builders['EMU:15Q:LOGICAL_EARLY'] = partial(
            self.build_logical_backend,
            n_qubits=15,
            distance=13,
            kappa_1=100,
            kappa_2=100_000,
            average_nb_photons=7,
            name='EMU:15Q:LOGICAL_EARLY',
        )
        self._backend_builders['EMU:1Q:LESCANNE_2020'] = partial(
            self.build_from_serialized,
            file_path=str(_PARENT_DIR / 'resources' / 'lescanne_2020.json'),
            name='EMU:1Q:LESCANNE_2020',
            average_nb_photons=4,
        )

    # pylint: disable=arguments-differ
    def backends(self, name: Optional[str] = None) -> List[BackendV2]:
        """Return a list of backends.

        Args:
            name (str): if provided, only backends with this name will be
                returned (there should be only one such backend).

        Returns:
            List[BackendV2]: the list of matching backends.
        """
        backends = [func() for func in self._backend_builders.values()]
        if name:
            return [backend for backend in backends if backend.name == name]
        return backends

    # Overloads for specific backend types
    @overload
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def get_backend(
        self,
        name: Literal['EMU:6Q:PHYSICAL_CATS', 'EMU:40Q:PHYSICAL_CATS'],
        *,
        n_qubits: Optional[int] = None,
        kappa_1: Optional[float] = None,
        kappa_2: Optional[float] = None,
        average_nb_photons: Optional[float] = None,
        clock_cycle: Optional[float] = None,
        coupling_map: Optional[List[Tuple[int, int]]] = None,
    ) -> ProcessorSimulator:
        """Get a physical cat backend."""

    @overload
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def get_backend(
        self,
        name: Literal['EMU:40Q:LOGICAL_TARGET', 'EMU:15Q:LOGICAL_EARLY'],
        *,
        n_qubits: Optional[int] = None,
        distance: Optional[int] = None,
        kappa_1: Optional[float] = None,
        kappa_2: Optional[float] = None,
        average_nb_photons: Optional[float] = None,
        clock_cycle: Optional[float] = None,
    ) -> ProcessorSimulator:
        """Get a logical cat backend."""

    @overload
    # pylint: disable=signature-differs
    def get_backend(
        self,
        name: Literal['EMU:40Q:LOGICAL_NOISELESS'],
        *,
        n_qubits: Optional[int] = None,
        clock_cycle: Optional[float] = None,
        **processor_kwargs,
    ) -> ProcessorSimulator:
        """Get a logical noiseless backend."""

    @overload
    def get_backend(
        self,
        name: Literal['EMU:1Q:LESCANNE_2020'],
        *,
        clock_cycle: Optional[float] = None,
        average_nb_photons: Optional[float] = None,
    ) -> ProcessorSimulator:
        """Get a serialized backend."""

    @overload
    # pylint: disable=signature-differs,arguments-differ
    def get_backend(
        self,
        name: str,
        **processor_kwargs,
    ) -> ProcessorSimulator:
        """Get any backend by name with arbitrary kwargs."""

    # pylint: disable=signature-differs,arguments-differ
    def get_backend(self, name: str, **processor_kwargs) -> ProcessorSimulator:
        """Get a backend by name with optional processor-specific parameters.

        Args:
            name: The name of the backend to retrieve
            **processor_kwargs: Backend-specific parameters that will be passed
                to the appropriate builder method

        Returns:
            ProcessorSimulator: The requested backend instance

        Raises:
            KeyError: If the backend name is not found
        """
        return self._backend_builders[name](**processor_kwargs)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def build_physical_backend(
        self,
        n_qubits: int = 5,
        kappa_1: float = 100,
        kappa_2: float = 10_000_000,
        average_nb_photons: float = 16,
        clock_cycle: float = 1e-9,
        coupling_map: Optional[List[Tuple[int, int]]] = None,
        name: Optional[str] = None,
    ) -> ProcessorSimulator:
        """Build a backend simulating a quantum processor made of physical cat
        qubits.

        Please refer to the docstring of ``PhysicalCatProcessor`` to learn more
        about the arguments.
        """
        return ProcessorSimulator(
            processor=PhysicalCatProcessor(
                n_qubits=n_qubits,
                kappa_1=kappa_1,
                kappa_2=kappa_2,
                average_nb_photons=average_nb_photons,
                clock_cycle=clock_cycle,
                coupling_map=coupling_map,
            ),
            name=name,
        )

    def build_logical_noiseless_backend(
        self,
        n_qubits: int = 40,
        clock_cycle: float = 1e-9,
        name: Optional[str] = None,
        **processor_kwargs,
    ) -> ProcessorSimulator:
        return ProcessorSimulator(
            processor=LogicalCatProcessor.create_noiseless(
                n_qubits=n_qubits,
                clock_cycle=clock_cycle,
                **processor_kwargs,
            ),
            translation_stage_plugin='sk_synthesis',
            name=name,
        )

    def build_custom_backend(
        self,
        backend_parameters: Optional[dict[str, Any]] = None,
        noise_models: Optional[dict[str, NoiseFunction]] = None,
        time_models: Optional[dict[str, TimeFunction]] = None,
        default_1q_noise_model: Optional[NoiseFunction] = None,
        default_1q_time_model: Optional[TimeFunction] = None,
        validate_parameters: Callable[
            [dict[str, float]], bool
        ] = lambda _: True,
        name: Optional[str] = None,
    ):
        if backend_parameters is None:
            backend_parameters = {
                'n_qubits': 15,
                'clock_cycle': 1e-9,
                'kappa_1': 100,
                'kappa_2': 10_000_000,
                'average_nb_photons': 16,
            }
        if noise_models is None:
            noise_models = {}
        if time_models is None:
            time_models = {}
        return ProcessorSimulator(
            processor=CustomCat(
                backend_parameters=backend_parameters,
                noise_models=noise_models,
                time_models=time_models,
                default_1q_noise_model=default_1q_noise_model,
                default_1q_time_model=default_1q_time_model,
                validate_parameters=validate_parameters,
            ),
            translation_stage_plugin='sk_synthesis',
            name=name,
        )

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def build_logical_backend(
        self,
        n_qubits: int = 40,
        distance: int = 15,
        kappa_1: float = 100,
        kappa_2: float = 10_000_000,
        average_nb_photons: float = 19,
        clock_cycle: float = 1e-9,
        name: Optional[str] = None,
    ) -> ProcessorSimulator:
        """Build a backend simulating a logical quantum processor whose logical
        qubits are made of physical cat qubits.

        Please refer to the docstring of ``LogicalCatProcessor`` to learn more
        about the arguments.
        """
        return ProcessorSimulator(
            processor=LogicalCatProcessor.create_noisy(
                n_qubits=n_qubits,
                distance=distance,
                kappa_1=kappa_1,
                kappa_2=kappa_2,
                average_nb_photons=average_nb_photons,
                clock_cycle=clock_cycle,
            ),
            translation_stage_plugin='sk_synthesis',
            name=name,
        )

    def build_from_serialized(
        self,
        file_path: str,
        clock_cycle: float = 1e-9,
        average_nb_photons: float = 16,
        name: Optional[str] = None,
    ) -> ProcessorSimulator:
        """Build a backend simulating a cat qubit based quantum processor
        described in a serialized format in a json file.

        See file ``lescanne_2020.json`` in this package as an example."""
        with open(file_path, encoding='utf-8') as f:
            serialized = SerializedProcessor.model_validate_json(f.read())
        return ProcessorSimulator(
            processor=InterpolatedCatProcessor(
                serialized_processor=serialized,
                clock_cycle=clock_cycle,
                alpha=np.sqrt(average_nb_photons),
            ),
            name=name,
        )
