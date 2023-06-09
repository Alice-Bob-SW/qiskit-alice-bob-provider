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

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit_alice_bob_provider.ensure_preparation_pass import (
    EnsurePreparationPass,
)

_pm = PassManager([EnsurePreparationPass()])


def test_missing_prep() -> None:
    c = QuantumCircuit(3, 1)
    c.x(0)
    c.reset(1)
    c.y(1)
    c.initialize('+', 2)
    c.cnot(1, 2)
    c.measure(2, 0)
    new_c = _pm.run(c)
    expected = c.count_ops()
    expected['reset'] += 1
    assert dict(new_c.count_ops()) == dict(expected)
