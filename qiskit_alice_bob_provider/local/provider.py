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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from qiskit.providers import BackendV2, ProviderV1

from ..processor.interpolated_cat import InterpolatedCatProcessor
from ..processor.physical_cat import PhysicalCatProcessor
from ..processor.serialization.model import SerializedProcessor
from .backend import ProcessorSimulator
from .coupling_maps import circular_map, rectangular_map

_PARENT_DIR = Path(__file__).parent


class AliceBobLocalProvider(ProviderV1):
    """Class listing a number of preset backends simulating cat qubit based
    quantum processors.

    Simulations will run locally on the machine where the backend is invoked.
    """

    def __init__(self) -> None:
        self._backends: List[ProcessorSimulator] = []
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
        self._backend_builders['EMU:1Q:LESCANNE_2020'] = partial(
            self.build_from_serialized,
            file_path=str(_PARENT_DIR / 'resources' / 'lescanne_2020.json'),
            name='EMU:1Q:LESCANNE_2020',
            average_nb_photons=4,
        )
        self._backends = [func() for func in self._backend_builders.values()]

    # pylint: disable=arguments-differ
    def backends(self, name: Optional[str] = None) -> List[BackendV2]:
        """Return a list of backends.

        Args:
            name (str): if provided, only backends with this name will be
                returned (there should be only one such backend).

        Returns:
            List[BackendV2]: the list of maching backends.
        """
        if name:
            return [
                backend for backend in self._backends if backend.name == name
            ]
        return self._backends

    # pylint: disable=signature-differs
    def get_backend(self, name: str, **processor_kwargs) -> ProcessorSimulator:
        return self._backend_builders[name](**processor_kwargs)

    # pylint: disable=too-many-arguments
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
                alpha=np.sqrt(average_nb_photons),
                clock_cycle=clock_cycle,
                coupling_map=coupling_map,
            ),
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
