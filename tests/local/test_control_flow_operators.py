import numpy as np
import pytest
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)

from qiskit_alice_bob_provider import AliceBobLocalProvider

qreg = QuantumRegister(1, 'q')
creg = ClassicalRegister(1, 'c')


def get_backends():
    """
    Return a list of all the names of each available backends.
    This is for better logging when running the tests.
    """
    provider = AliceBobLocalProvider()
    return [backend.name for backend in provider.backends()]


def generate_basic_circuit():
    basic_circ = QuantumCircuit(qreg, creg)
    basic_circ.name = 'Basic'
    basic_circ.rz(np.pi, qreg[0])  # Equivalent to X gate
    basic_circ.measure(qreg[0], creg[0])
    return basic_circ


def generate_if_circuit():
    if_else_circ = QuantumCircuit(qreg, creg)
    if_else_circ.name = 'If_Else'
    if_else_circ.rz(np.pi, qreg[0])  # Equivalent to X gate
    if_else_circ.measure(qreg[0], creg[0])

    with if_else_circ.if_test((creg[0], 1)) as _else:
        if_else_circ.rz(np.pi, qreg[0])  # Apply RZ(pi) if measured 1
    with _else:
        if_else_circ.rz(np.pi, qreg[0])  # Apply RZ(pi) if measured 0
    if_else_circ.measure(qreg[0], creg[0])
    return if_else_circ


def generate_while_circuit():
    while_circ = QuantumCircuit(qreg, creg)
    while_circ.name = 'While'
    while_circ.rz(np.pi, qreg[0])  # Ensure q[0] is initially |1>
    while_circ.measure(qreg[0], creg[0])

    with while_circ.while_loop((creg[0], 1)):
        while_circ.rz(np.pi, qreg[0])
        while_circ.measure(qreg[0], creg[0])
    return while_circ


def generate_for_circuit():
    for_circ = QuantumCircuit(qreg, creg)
    for_circ.name = 'For'
    with for_circ.for_loop(range(3)):
        for_circ.rz(np.pi, qreg[0])
    for_circ.measure(qreg[0], creg[0])
    return for_circ


def generate_switch_circuit():
    switch_circ = QuantumCircuit(qreg, creg)
    switch_circ.name = 'Switch'
    switch_circ.measure(qreg[0], creg[0])

    with switch_circ.switch(creg[0]) as case:
        with case(0):
            switch_circ.rz(np.pi, qreg[0])  # Apply RZ(pi) if measured 0
        with case(1):
            switch_circ.rz(np.pi, qreg[0])  # Apply RZ(pi) if measured 1
    switch_circ.measure(qreg[0], creg[0])
    return switch_circ


CIRCUITS = [
    generate_basic_circuit,
    generate_if_circuit,
    generate_while_circuit,
    generate_for_circuit,
    generate_switch_circuit,
]


@pytest.mark.parametrize('make_circ', CIRCUITS)
@pytest.mark.parametrize('backend_name', get_backends())
def test_circuit_runs_on_simulators(make_circ, backend_name):
    provider = AliceBobLocalProvider()
    backend = provider.get_backend(backend_name)
    circuit = make_circ()
    transpiled = transpile(circuit, backend)
    job = backend.run(transpiled, shots=10)
    result = job.result()
    assert result.success is True

    if circuit.name == 'If_Else':
        assert any(instr for instr in circuit.data if instr.name == 'if_else')
    elif circuit.name == 'While':
        assert any(
            instr for instr in circuit.data if instr.name == 'while_loop'
        )
    elif circuit.name == 'For':
        assert any(instr for instr in circuit.data if instr.name == 'for_loop')
    elif circuit.name == 'Switch':
        assert any(
            instr for instr in circuit.data if instr.name == 'switch_case'
        )
