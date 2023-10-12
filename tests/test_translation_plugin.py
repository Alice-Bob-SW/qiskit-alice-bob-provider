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
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Reset
from qiskit.extensions.quantum_initializer import Initialize
from qiskit.transpiler import PassManager

from qiskit_alice_bob_provider.errors import AliceBobTranspilationException
from qiskit_alice_bob_provider.remote.translation_plugin import (
    StatePreparationPass,
)

_pm = PassManager([StatePreparationPass()])


def test_str_plus() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize('+', 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, np.array([1, 1]) / np.sqrt(2))


def test_str_minus() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize('-', 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, np.array([1, -1]) / np.sqrt(2))


def test_str_0() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize('0', 1)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('reset')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 1
    op = instructions[0].operation
    assert isinstance(op, Reset)


def test_str_1() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize('1', 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, [0, 1])


def test_int_0() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize(0, 1)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('reset')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 1
    op = instructions[0].operation
    assert isinstance(op, Reset)


def test_int_1() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize(1, 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, [0, 1])


def test_complex_plus() -> None:
    c = QuantumCircuit(2, 1)
    state = np.array([1, 1]) / np.sqrt(2)
    c.initialize(state, 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, state)


def test_complex_plus_with_global_phase() -> None:
    c = QuantumCircuit(2, 1)
    state = np.array([1, 1]) / np.sqrt(2) * np.exp(3j)
    c.initialize(state, 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, state)


def test_complex_minus() -> None:
    c = QuantumCircuit(2, 1)
    state = np.array([1, -1]) / np.sqrt(2)
    c.initialize(state, 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, state)


def test_complex_0() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize([1, 0], 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('reset')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Reset)


def test_complex_1() -> None:
    c = QuantumCircuit(2, 1)
    c.initialize([0, 1], 0)
    new_c = _pm.run(c)
    instructions = new_c.get_instructions('initialize')
    assert len(instructions) == 1
    assert c.find_bit(instructions[0].qubits[0]).index == 0
    op = instructions[0].operation
    assert isinstance(op, Initialize)
    assert np.array_equal(op.params, [0, 1])


def test_complex_unsupported() -> None:
    c = QuantumCircuit(2, 1)
    state = np.array([1, -3j])
    state /= np.sqrt(np.sum(state * np.conj(state)))
    c.initialize(state, 0)
    with pytest.raises(AliceBobTranspilationException):
        _pm.run(c)
