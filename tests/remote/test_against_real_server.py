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

import pytest
from qiskit import QuantumCircuit, execute

from qiskit_alice_bob_provider import AliceBobRemoteProvider

realserver = pytest.mark.skipif(
    "not config.getoption('api_key') and not config.getoption('base_url')"
)


@realserver
def test_happy_path(base_url: str, api_key: str) -> None:
    c = QuantumCircuit(1, 2)
    c.initialize('+', 0)
    c.measure_x(0, 0)
    c.measure(0, 1)
    provider = AliceBobRemoteProvider(
        api_key=api_key,
        url=base_url,
    )
    backend = provider.get_backend('EMU:1Q:LESCANNE_2020')
    job = execute(c, backend)
    res = job.result()
    res.get_counts()
