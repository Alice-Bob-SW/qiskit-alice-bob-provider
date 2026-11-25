import warnings
from typing import Dict, List

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Delay, Parameter
from qiskit.transpiler.exceptions import TranspilerError
from requests_mock.mocker import Mocker

from qiskit_alice_bob_provider.qir import circuit_to_qir
from qiskit_alice_bob_provider.remote.provider import AliceBobRemoteProvider

_NAMES = ['EMU:1Q:LESCANNE_2020', 'ALL_INSTRUCTIONS', 'EMU:18Q:HELIUM']
_LEVELS = [0, 1, 2, 3]


@pytest.fixture
def provider(
    targets: List[Dict], helium_target: Dict, requests_mock: Mocker
) -> AliceBobRemoteProvider:
    requests_mock.register_uri(
        'GET', '/v1/targets/', json=[*targets, helium_target]
    )
    return AliceBobRemoteProvider(api_key='foo')


def _delays(circuit: QuantumCircuit) -> List[Delay]:
    delays = []
    for instruction in circuit.data:
        if isinstance(instruction.operation, Delay):
            delays.append(instruction.operation)
    return delays


def _transpile_delay(
    provider: AliceBobRemoteProvider,
    name: str,
    level: int,
    duration: float,
    unit: str,
) -> Delay:
    c = QuantumCircuit(1, 1)
    c.delay(duration, 0, unit=unit)
    c.measure(0, 0)
    backend = provider.get_backend(name)
    (delay,) = _delays(transpile(c, backend, optimization_level=level))
    return delay


@pytest.mark.parametrize('name', _NAMES)
@pytest.mark.parametrize('level', _LEVELS)
@pytest.mark.parametrize(
    'duration, unit, nanoseconds',
    [
        (2, 's', 2_000_000_000),
        (3, 'ms', 3_000_000),
        (1.5, 'us', 1500),
        (3000, 'ns', 3000),
        (0, 'ns', 0),
        (2000, 'ps', 2),
        (0.013, 'us', 13),
        (0.0101, 'ms', 10100),
        (0.30000000000000004, 'us', 300),
    ],
)
def test_transpile_converts_each_delay_to_whole_nanoseconds(
    provider: AliceBobRemoteProvider,
    name: str,
    level: int,
    duration: float,
    unit: str,
    nanoseconds: int,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Rounded')
        delay = _transpile_delay(provider, name, level, duration, unit)
    assert (delay.duration, delay.unit) == (nanoseconds, 'ns')
    assert isinstance(delay.duration, int)


@pytest.mark.parametrize('level', _LEVELS)
@pytest.mark.parametrize(
    'duration, unit, nanoseconds',
    [
        (7.5, 'ns', 8),
        (2.5, 'ns', 2),
        (1500, 'ps', 2),
        (0.0011, 'us', 1),
        (0.0125, 'us', 12),
        (0.0105, 'us', 10),
    ],
)
def test_transpile_rounds_a_half_to_even_and_warns(
    provider: AliceBobRemoteProvider,
    level: int,
    duration: float,
    unit: str,
    nanoseconds: int,
) -> None:
    with pytest.warns(UserWarning) as record:
        delay = _transpile_delay(
            provider, 'EMU:18Q:HELIUM', level, duration, unit
        )
    assert (delay.duration, delay.unit) == (nanoseconds, 'ns')
    rounded = [str(w.message) for w in record if 'Rounded' in str(w.message)]
    assert len(rounded) == 1
    assert f'to {nanoseconds} ns' in rounded[0]


@pytest.mark.parametrize('level', _LEVELS)
@pytest.mark.parametrize(
    'duration, unit, nanoseconds',
    [(1.5, 'us', 1500), (0.012, 'us', 12), (3, 'ms', 3_000_000)],
)
def test_default_scheduling_method_runs_a_whole_delay(
    provider: AliceBobRemoteProvider,
    level: int,
    duration: float,
    unit: str,
    nanoseconds: int,
) -> None:
    c = QuantumCircuit(1, 1)
    c.delay(duration, 0, unit=unit)
    c.measure(0, 0)
    backend = provider.get_backend('EMU:18Q:HELIUM')
    transpiled = transpile(
        c, backend, optimization_level=level, scheduling_method='default'
    )
    assert f'@__quantum__qis__delay__body(i64 {nanoseconds},' in (
        circuit_to_qir(transpiled)
    )


def test_default_scheduling_method_refuses_a_fractional_delay(
    provider: AliceBobRemoteProvider,
) -> None:
    c = QuantumCircuit(1, 1)
    c.delay(0.0125, 0, unit='us')
    c.measure(0, 0)
    backend = provider.get_backend('EMU:18Q:HELIUM')
    transpiled = transpile(c, backend, scheduling_method='default')
    with pytest.raises(ValueError, match='as a whole number of nanoseconds'):
        circuit_to_qir(transpiled)


@pytest.mark.parametrize('level', _LEVELS)
def test_transpile_refuses_a_delay_in_dt(
    provider: AliceBobRemoteProvider, level: int
) -> None:
    c = QuantumCircuit(1)
    c.delay(100, 0, unit='dt')
    backend = provider.get_backend('EMU:18Q:HELIUM')
    with pytest.raises(TranspilerError, match='a unit of time'):
        transpile(c, backend, optimization_level=level)


@pytest.mark.parametrize('level', _LEVELS)
@pytest.mark.parametrize(
    'unit, value, nanoseconds',
    [('us', 1.5, 1500), ('us', 1.001, 1001), ('s', 3.3e-7, 330)],
)
def test_delay_bound_after_transpilation_runs_in_nanoseconds(
    provider: AliceBobRemoteProvider,
    level: int,
    unit: str,
    value: float,
    nanoseconds: int,
) -> None:
    c = QuantumCircuit(1, 1)
    t = Parameter('t')
    c.delay(t, 0, unit=unit)
    c.measure(0, 0)
    backend = provider.get_backend('EMU:18Q:HELIUM')
    transpiled = transpile(c, backend, optimization_level=level)
    (delay,) = _delays(transpiled)
    assert delay.unit == 'ns'
    qir = circuit_to_qir(transpiled.assign_parameters({t: value}))
    assert f'@__quantum__qis__delay__body(i64 {nanoseconds},' in qir


def test_delay_bound_to_a_fraction_of_a_nanosecond_is_refused(
    provider: AliceBobRemoteProvider,
) -> None:
    c = QuantumCircuit(1, 1)
    t = Parameter('t')
    c.delay(t, 0, unit='us')
    c.measure(0, 0)
    backend = provider.get_backend('EMU:18Q:HELIUM')
    transpiled = transpile(c, backend, optimization_level=0)
    with pytest.raises(ValueError, match='after you bind its parameters'):
        circuit_to_qir(transpiled.assign_parameters({t: 0.0011}))
