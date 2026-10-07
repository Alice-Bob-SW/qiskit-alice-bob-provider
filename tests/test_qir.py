from textwrap import dedent

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.classical import expr

import qiskit_alice_bob_provider  # noqa: F401
from qiskit_alice_bob_provider.qir import circuit_to_qir

Q0 = '%Qubit* null'
Q1 = '%Qubit* inttoptr (i64 1 to %Qubit*)'
R0 = '%Result* null'
R1 = '%Result* inttoptr (i64 1 to %Result*)'


def _call(name: str, *args: str) -> str:
    return f'call void @__quantum__qis__{name}__body({", ".join(args)})'


def test_measure_x_emits_mx() -> None:
    c = QuantumCircuit(1, 1)
    c.measure_x(0, 0)
    assert _call('mx', Q0, R0) in circuit_to_qir(c)


@pytest.mark.parametrize(
    'state, basis, value',
    [
        ('0', 'z', 'false'),
        ('1', 'z', 'true'),
        ('+', 'x', 'false'),
        ('-', 'x', 'true'),
    ],
)
def test_initialize_emits_prepare(state: str, basis: str, value: str) -> None:
    c = QuantumCircuit(1)
    c.initialize(state, 0)
    assert _call(f'prepare_{basis}', f'i1 {value}', Q0) in circuit_to_qir(c)


@pytest.mark.parametrize(
    'duration, unit, microseconds',
    [
        (2.0, 's', '2.000000e+06'),
        (3.0, 'ms', '3.000000e+03'),
        (1.5, 'us', '1.500000e+00'),
        (100, 'us', '1.000000e+02'),
        (3000, 'ns', '3.000000e+00'),
        (2000, 'ps', '2.000000e-03'),
    ],
)
def test_delay_emits_microseconds(
    duration: float, unit: str, microseconds: str
) -> None:
    c = QuantumCircuit(1)
    c.delay(duration, 0, unit=unit)
    assert _call('delay', f'double {microseconds}', Q0) in circuit_to_qir(c)


def test_delay_in_dt_raises() -> None:
    c = QuantumCircuit(2, 1)
    c.measure(0, 0)
    with c.if_test((c.clbits[0], 1)):
        c.delay(100, 1, unit='dt')
    with pytest.raises(ValueError, match=r'qubits \[1\] has the unit "dt"'):
        circuit_to_qir(c)


def test_each_intrinsic_is_declared_once() -> None:
    c = QuantumCircuit(1, 2)
    c.initialize('+', 0)
    c.initialize('-', 0)
    c.delay(1.0, 0, unit='us')
    c.delay(2.0, 0, unit='us')
    c.measure_x(0, 0)
    c.measure_x(0, 1)
    qir = circuit_to_qir(c)
    for name in ('prepare_x', 'delay', 'mx'):
        assert qir.count(f'declare void @__quantum__qis__{name}__body') == 1


@pytest.mark.parametrize(
    'gate, params, num_qubits, intrinsic',
    [
        ('ccx', [], 3, 'ccx__body'),
        ('cx', [], 2, 'cnot__body'),
        ('cz', [], 2, 'cz__body'),
        ('h', [], 1, 'h__body'),
        ('reset', [], 1, 'reset__body'),
        ('rx', [0.5], 1, 'rx__body'),
        ('ry', [0.5], 1, 'ry__body'),
        ('rz', [0.5], 1, 'rz__body'),
        ('s', [], 1, 's__body'),
        ('sdg', [], 1, 's__adj'),
        ('swap', [], 2, 'swap__body'),
        ('t', [], 1, 't__body'),
        ('tdg', [], 1, 't__adj'),
        ('x', [], 1, 'x__body'),
        ('y', [], 1, 'y__body'),
        ('z', [], 1, 'z__body'),
    ],
)
def test_standard_gate_emits_intrinsic(
    gate: str, params: list, num_qubits: int, intrinsic: str
) -> None:
    c = QuantumCircuit(num_qubits)
    getattr(c, gate)(*params, *range(num_qubits))
    assert f'@__quantum__qis__{intrinsic}(' in circuit_to_qir(c)


def test_id_emits_two_x() -> None:
    c = QuantumCircuit(1)
    c.id(0)
    assert circuit_to_qir(c).count(_call('x', Q0)) == 2


def test_gate_with_definition_emits_its_definition() -> None:
    definition = QuantumCircuit(2)
    definition.h(1)
    gate = Gate('custom', 2, [])
    gate.definition = definition
    c = QuantumCircuit(2)
    c.append(gate, [0, 1])
    assert _call('h', Q1) in circuit_to_qir(c)


def test_if_else_on_bit_emits_both_branches() -> None:
    c = QuantumCircuit(1, 1)
    with c.if_test((c.clbits[0], 1)) as else_:
        c.x(0)
    with else_:
        c.z(0)
    expected = f"""\
          br i1 %0, label %then, label %else

        then:                                             ; preds = %entry
          {_call('x', Q0)}
          br label %continue

        else:                                             ; preds = %entry
          {_call('z', Q0)}
    """
    assert dedent(expected) in circuit_to_qir(c)


def test_if_else_on_register_reads_each_bit() -> None:
    creg = ClassicalRegister(2)
    c = QuantumCircuit(QuantumRegister(1), creg)
    with c.if_test((creg, 2)):
        c.x(0)
    qir = circuit_to_qir(c)
    for result in (R0, R1):
        assert f'@__quantum__qis__read_result__body({result})' in qir
    assert qir.count(_call('x', Q0)) == 1


def test_if_else_value_larger_than_register_raises() -> None:
    creg = ClassicalRegister(1)
    c = QuantumCircuit(QuantumRegister(1), creg)
    with c.if_test((creg, 2)):
        c.x(0)
    with pytest.raises(ValueError, match='too large'):
        circuit_to_qir(c)


def test_if_else_on_expression_raises() -> None:
    c = QuantumCircuit(1, 1)
    with c.if_test(expr.logic_not(c.clbits[0])):
        c.x(0)
    with pytest.raises(ValueError, match='one bit or on one register'):
        circuit_to_qir(c)


def test_output_is_recorded_per_register() -> None:
    c = QuantumCircuit(
        QuantumRegister(1), ClassicalRegister(2), ClassicalRegister(1)
    )
    expected = [
        'array_record_output(i64 2, i8* null)',
        'result_record_output(%Result* inttoptr (i64 1 to %Result*), i8* null)',
        'result_record_output(%Result* null, i8* null)',
        'array_record_output(i64 1, i8* null)',
        'result_record_output(%Result* inttoptr (i64 2 to %Result*), i8* null)',
    ]
    records = '\n'.join(f'  call void @__quantum__rt__{r}' for r in expected)
    assert records in circuit_to_qir(c)


def test_unsupported_state_raises() -> None:
    c = QuantumCircuit(2)
    c.initialize('01')
    with pytest.raises(ValueError, match='one qubit at a time'):
        circuit_to_qir(c)


def test_unsupported_instruction_raises() -> None:
    c = QuantumCircuit(1)
    c.append(Instruction('foo', 1, 0, []), [0])
    with pytest.raises(ValueError, match='"foo"'):
        circuit_to_qir(c)
