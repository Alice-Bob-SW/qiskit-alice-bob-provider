import sys
from typing import List

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Instruction
from qiskit.circuit.library import Initialize, get_standard_gate_name_mapping

from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.processor.logical_cat import LogicalCatProcessor
from qiskit_alice_bob_provider.processor.physical_cat import (
    PhysicalCatProcessor,
)

from .processor_fixture import SimpleAllToAllProcessor, SimpleProcessor


def _assert_one_initialize(circuit: QuantumCircuit, expected: str) -> None:
    initializes = circuit.get_instructions('initialize')
    assert len(initializes) == 1
    assert isinstance(initializes[0].operation, Initialize)
    params = initializes[0].operation.params
    assert params == list(expected)


def _assert_many_initializes(
    circuit: QuantumCircuit, expected: List[str]
) -> None:
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
    backend.set_options(memory=True)
    assert backend.options['shots'] == 7
    transpiled = transpile(circ, backend)
    run = backend.run(transpiled, memory=True)
    assert sum(run.result().get_counts().values()) == 7
    assert len(run.result().get_memory()) == 7


def test_run_override_nbar_error() -> None:
    """
    Test that passing average_nb_photons option to backend.run() is not allowed
    """
    circ = QuantumCircuit(1, 1)
    backend = ProcessorSimulator(PhysicalCatProcessor())
    transpiled = transpile(circ, backend)
    with pytest.raises(ValueError):
        _ = backend.run(transpiled, shots=100, average_nb_photons=4)
    with pytest.raises(ValueError):
        _ = backend.run(transpiled, shots=100, kappa_1=4)

    backend = ProcessorSimulator(LogicalCatProcessor())
    transpiled = transpile(circ, backend)
    with pytest.raises(ValueError):
        _ = backend.run(transpiled, shots=100, average_nb_photons=4)
    with pytest.raises(ValueError):
        _ = backend.run(transpiled, shots=100, kappa_1=4)


def test_translation_plugin() -> None:
    backend = ProcessorSimulator(SimpleProcessor(1))

    # if Initialize('+'), do nothing
    circ = QuantumCircuit(1)
    circ.initialize('+')
    transpiled = transpile(circ, backend)
    _assert_one_initialize(transpiled, '+')

    # if no reset / initialize, add Initialize('0')
    circ = QuantumCircuit(1)
    circ.x(0)
    transpiled = transpile(circ, backend)
    _assert_one_initialize(transpiled, '0')

    # if reset, convert to Intialize('0')
    circ = QuantumCircuit(1)
    circ.reset(0)
    transpiled = transpile(circ, backend)
    _assert_one_initialize(transpiled, '0')

    # if reset, convert to Intialize('0')
    circ = QuantumCircuit(3)
    circ.initialize(2, [0, 1])
    circ.reset(2)

    # Seed fixed to avoid qubit shuffling from VF2Layout transpiler pass,
    # needed for a deterministic test
    transpiled = transpile(circ, backend, seed_transpiler=4)
    _assert_many_initializes(transpiled, ['0', '1', '0'])


def test_synthesize_rz() -> None:
    backend = ProcessorSimulator(
        SimpleAllToAllProcessor(), translation_stage_plugin='sk_synthesis'
    )
    circ = QuantumCircuit(1)
    circ.rz(np.pi * 0.25, 0)
    transpiled = transpile(circ, backend)
    assert len(transpiled.get_instructions('rz')) == 0
    assert len(transpiled.get_instructions('t')) == 1


def test_synthesize_cz() -> None:
    backend = ProcessorSimulator(
        SimpleAllToAllProcessor(), translation_stage_plugin='sk_synthesis'
    )
    circ = QuantumCircuit(2)
    circ.cz(0, 1)
    transpiled = transpile(circ, backend)
    assert len(transpiled.get_instructions('cz')) == 0
    assert len(transpiled.get_instructions('h')) == 2


def test_all_gates():
    """Test transpilation for all basis gates"""
    provider = AliceBobLocalProvider()
    backend = provider.get_backend('EMU:40Q:LOGICAL_TARGET')

    qiskit_gates = get_standard_gate_name_mapping()
    skip_gates = [
        # Trying to access this gates from the circuit attributes returns an
        # error "'QuantumCircuit' object has no attribute '...'"
        'c3sx',
        'cu1',
        'cu3',
        'xx_minus_yy',
        'xx_plus_yy',
        'u1',
        'u2',
        'u3',
        # This one is not a gate, just a float.
        'global_phase',
    ]
    if sys.platform == 'darwin':
        # For some reason, on macOS we have numerical instabilities with the
        # Solovay Kitaev synthesis, with specific angles.
        # For instance, a simple 1Q circuit with a RZ(5pi/4) gate fails to
        # transpile with our logical backends, and typically raises :
        #   ValueError('Input matrix is not orthogonal.')
        # As a result, the synthesis currently fails for the gates below (this
        # needs to be fixed).
        skip_gates += [
            'cry',
            'rccx',
            'rcccx',
        ]

    def create_circuit_with_gate(instruction: Instruction):
        if instruction.params:
            # Most parameters are angles -> use pi/5.
            # Except for the delay instruction, which expects an integer for
            # param 't' -> use 10.
            params = [
                10 if p.name == 't' else np.pi / 5 for p in instruction.params
            ]
        else:
            params = []
        circuit = QuantumCircuit(
            instruction.num_qubits, instruction.num_clbits
        )
        args = (
            params
            + list(range(instruction.num_qubits))
            + list(range(instruction.num_clbits))
        )
        # apply the gate
        getattr(circuit, instruction.name)(*args)
        return circuit

    errors = []
    for name, i in qiskit_gates.items():
        if name in skip_gates:
            continue
        try:
            circ = create_circuit_with_gate(i)
            _ = transpile(circ, backend=backend)
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append((name, e))
    assert not errors


def test_do_nothing_on_mx() -> None:
    backend = ProcessorSimulator(
        SimpleAllToAllProcessor(), translation_stage_plugin='sk_synthesis'
    )
    circ = QuantumCircuit(1, 1)
    circ.measure_x(0, 0)
    transpiled = transpile(circ, backend)
    assert len(transpiled.get_instructions('measure_x')) == 1


def test_do_nothing_on_pp() -> None:
    backend = ProcessorSimulator(
        SimpleAllToAllProcessor(), translation_stage_plugin='sk_synthesis'
    )
    circ = QuantumCircuit(1)
    circ.initialize('+', 0)
    transpiled = transpile(circ, backend)
    assert len(transpiled.get_instructions('initialize')) == 1
