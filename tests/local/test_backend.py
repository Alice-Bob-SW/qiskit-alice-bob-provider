from typing import List

from qiskit import QuantumCircuit, execute, transpile
from qiskit.extensions.quantum_initializer import Initialize

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.processor.physical_cat import (
    PhysicalCatProcessor,
)

from .processor_fixture import SimpleProcessor


def _check_initialize(circuit: QuantumCircuit, expected: str) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == 1
    assert isinstance(initializes[0].operation, Initialize)
    params = initializes[0].operation.params
    assert params == list(expected)


def _check_initializes(circuit: QuantumCircuit, expected: List[str]) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == len(expected)
    for initialize in initializes:
        assert len(initialize.qubits) == 1
        qubit = circuit.find_bit(initialize.qubits[0]).index
        assert isinstance(initialize.operation, Initialize)
        assert initialize.operation.params[0] == expected[qubit]


def test_set_execution_backend_options() -> None:
    circ = QuantumCircuit(1, 1)
    circ.initialize('+')
    circ.measure_x(0, 0)
    backend = ProcessorSimulator(PhysicalCatProcessor())
    backend.set_options(shots=7)
    assert backend.options['shots'] == 7
    assert sum(execute(circ, backend).result().get_counts().values()) == 7


def test_translation_plugin() -> None:
    backend = ProcessorSimulator(SimpleProcessor(1))

    # if Initialize('+'), do nothing
    circ = QuantumCircuit(1)
    circ.initialize('+')
    transpiled = transpile(circ, backend)
    _check_initialize(transpiled, '+')

    # if no reset / initialize, add Initialize('0')
    circ = QuantumCircuit(1)
    circ.x(0)
    transpiled = transpile(circ, backend)
    _check_initialize(transpiled, '0')

    # if reset, convert to Intialize('0')
    circ = QuantumCircuit(1)
    circ.reset(0)
    transpiled = transpile(circ, backend)
    _check_initialize(transpiled, '0')

    # if reset, convert to Intialize('0')
    circ = QuantumCircuit(3)
    circ.initialize(2, [0, 1])
    circ.reset(2)
    transpiled = transpile(circ, backend)
    _check_initializes(transpiled, ['0', '1', '0'])
