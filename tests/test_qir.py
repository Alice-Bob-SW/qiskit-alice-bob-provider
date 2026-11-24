from textwrap import dedent

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, Parameter
from qiskit.circuit.classical import expr

import qiskit_alice_bob_provider  # noqa: F401
from qiskit_alice_bob_provider.qir import circuit_to_qir

Q0 = 'ptr null'
Q1 = 'ptr inttoptr (i64 1 to ptr)'
R0 = 'ptr null'
R1 = 'ptr inttoptr (i64 1 to ptr)'


def _call(name: str, *args: str) -> str:
    return f'call void @__quantum__qis__{name}__body({", ".join(args)})'


def test_sx_emits_sx() -> None:
    c = QuantumCircuit(1)
    c.sx(0)
    qir = circuit_to_qir(c)
    assert _call('sx', Q0) in qir
    assert '__quantum__qis__h__body' not in qir
    assert '__quantum__qis__s__' not in qir


def test_measure_x_emits_mx() -> None:
    c = QuantumCircuit(1, 1)
    c.measure_x(0, 0)
    assert _call('mx', Q0, R0) in circuit_to_qir(c)


def test_measure_y_emits_my() -> None:
    c = QuantumCircuit(1, 1)
    c.measure_y(0, 0)
    assert _call('my', Q0, R0) in circuit_to_qir(c)


@pytest.mark.parametrize(
    'state, basis, value',
    [
        ('0', 'z', 'false'),
        ('1', 'z', 'true'),
        ('+', 'x', 'false'),
        ('-', 'x', 'true'),
        ('r', 'y', 'false'),
        ('l', 'y', 'true'),
    ],
)
def test_initialize_emits_prepare(state: str, basis: str, value: str) -> None:
    c = QuantumCircuit(1)
    c.initialize(state, 0)
    assert _call(f'prepare_{basis}', f'i1 {value}', Q0) in circuit_to_qir(c)


@pytest.mark.parametrize(
    'duration, unit, nanoseconds',
    [
        (0, 'ns', 0),
        (8, 'ns', 8),
        (8.0, 'ns', 8),
        (1000.9999999999999, 'ns', 1001),
        (1.5, 'us', 1500),
        (2, 's', 2_000_000_000),
        (1.2000000000000002e-08, 's', 12),
        (2000, 'ps', 2),
        (2**63 - 1, 'ns', 2**63 - 1),
    ],
)
def test_delay_emits_whole_nanoseconds(
    duration: float, unit: str, nanoseconds: int
) -> None:
    c = QuantumCircuit(1)
    c.delay(duration, 0, unit=unit)
    qir = circuit_to_qir(c)
    assert _call('delay', f'i64 {nanoseconds}', Q0) in qir
    assert 'declare void @__quantum__qis__delay__body(i64, ptr)' in qir


@pytest.mark.parametrize(
    'duration, unit',
    [(7.5, 'ns'), (1.1, 'ns'), (1500, 'ps'), (0.0125, 'us')],
)
def test_delay_not_in_whole_nanoseconds_raises(
    duration: float, unit: str
) -> None:
    c = QuantumCircuit(1)
    c.delay(duration, 0, unit=unit)
    with pytest.raises(ValueError, match='as a whole number of nanoseconds'):
        circuit_to_qir(c)


@pytest.mark.parametrize('duration', [2**63, 2**64, 1e19])
def test_delay_longer_than_an_i64_raises(duration: float) -> None:
    c = QuantumCircuit(1)
    c.delay(duration, 0, unit='ns')
    with pytest.raises(ValueError, match='longer than the longest delay'):
        circuit_to_qir(c)


def test_delay_in_dt_names_the_unit() -> None:
    c = QuantumCircuit(1)
    c.delay(2, 0, unit='dt')
    with pytest.raises(ValueError, match='has the unit "dt", which has no'):
        circuit_to_qir(c)


def test_stretch_delay_raises_value_error() -> None:
    c = QuantumCircuit(1)
    c.delay(c.add_stretch('a'), 0)
    with pytest.raises(ValueError, match='no duration in nanoseconds'):
        circuit_to_qir(c)


def test_delay_error_names_the_qubit_inside_a_block() -> None:
    c = QuantumCircuit(2, 1)
    c.measure(0, 0)
    with c.if_test((c.clbits[0], 1)):
        c.delay(100, 1, unit='dt')
    with pytest.raises(ValueError, match=r'100 dt on qubits \[1\]'):
        circuit_to_qir(c)


def test_unbound_parameter_raises() -> None:
    c = QuantumCircuit(1, 1)
    c.measure(0, 0)
    with c.if_test((c.clbits[0], 1)):
        c.rz(Parameter('theta'), 0)
    with pytest.raises(ValueError, match=r"\['theta'\] are not bound"):
        circuit_to_qir(c)


def test_each_intrinsic_is_declared_once() -> None:
    c = QuantumCircuit(1, 2)
    c.initialize('+', 0)
    c.initialize('-', 0)
    c.delay(1, 0, unit='ns')
    c.delay(2, 0, unit='ns')
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
        'array_record_output(i64 2, ptr null)',
        'result_record_output(ptr inttoptr (i64 1 to ptr), ptr null)',
        'result_record_output(ptr null, ptr null)',
        'array_record_output(i64 1, ptr null)',
        'result_record_output(ptr inttoptr (i64 2 to ptr), ptr null)',
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
