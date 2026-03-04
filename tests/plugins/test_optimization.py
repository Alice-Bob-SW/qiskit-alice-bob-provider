from unittest.mock import MagicMock

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManagerConfig

from qiskit_alice_bob_provider import transpile
from qiskit_alice_bob_provider.local import AliceBobLocalProvider
from qiskit_alice_bob_provider.plugins.optimization import (
    AliceBobOptimizationPlugin,
)


def test_plugin_returns_none_for_level_0():
    config = PassManagerConfig()
    plugin = AliceBobOptimizationPlugin()
    assert plugin.pass_manager(config, optimization_level=0) is None


def test_plugin_returns_pass_manager_for_level_1():
    config = PassManagerConfig()
    plugin = AliceBobOptimizationPlugin()
    pm = plugin.pass_manager(config, optimization_level=1)
    assert pm is not None


def test_plugin_returns_pass_manager_for_default_level():
    config = PassManagerConfig()
    plugin = AliceBobOptimizationPlugin()
    pm = plugin.pass_manager(config)
    assert pm is not None


def test_transpile_with_optimization_plugin():
    """Test that transpilation succeeds with the ab_optimization plugin
    on an SK synthesis backend."""
    provider = AliceBobLocalProvider()
    backend = provider.get_backend('EMU:40Q:LOGICAL_NOISELESS')

    circ = QuantumCircuit(1, 1)
    circ.initialize(0, [0])
    circ.x(0)
    circ.x(0)
    circ.delay(10, 0)
    circ.measure_x(0, 0)

    transpiled = transpile(circ, backend)
    assert transpiled is not None


def test_convenience_transpile_auto_detects_plugin():
    """Test that the convenience transpile() auto-detects the optimization
    plugin from the backend."""
    provider = AliceBobLocalProvider()
    backend = provider.get_backend('EMU:40Q:LOGICAL_NOISELESS')

    assert backend.get_optimization_stage_plugin() == 'ab_optimization'

    circ = QuantumCircuit(1, 1)
    circ.initialize(0, [0])
    circ.x(0)
    circ.delay(10, 0)
    circ.measure_x(0, 0)

    transpiled = transpile(circ, backend)
    assert transpiled is not None


def test_convenience_transpile_respects_explicit_method():
    """Test that an explicitly passed optimization_method is not overridden."""
    provider = AliceBobLocalProvider()
    backend = provider.get_backend('EMU:40Q:LOGICAL_NOISELESS')

    circ = QuantumCircuit(1, 1)
    circ.initialize(0, [0])
    circ.x(0)
    circ.delay(10, 0)
    circ.measure_x(0, 0)

    # Passing ab_optimization explicitly should also work
    transpiled = transpile(
        circ, backend, optimization_method='ab_optimization'
    )
    assert transpiled is not None


def test_convenience_transpile_no_plugin_backend():
    """Test that the convenience transpile works for backends without
    get_optimization_stage_plugin."""
    provider = AliceBobLocalProvider()
    backend = provider.get_backend('EMU:6Q:PHYSICAL_CATS')

    assert backend.get_optimization_stage_plugin() is None

    circ = QuantumCircuit(1, 1)
    circ.initialize(0, [0])
    circ.x(0)
    circ.delay(10, 0)
    circ.measure(0, 0)

    transpiled = transpile(circ, backend)
    assert transpiled is not None


def test_logical_backend_has_optimization_plugin():
    """Test that logical backends are configured with the optimization
    plugin."""
    provider = AliceBobLocalProvider()

    noiseless = provider.get_backend('EMU:40Q:LOGICAL_NOISELESS')
    assert noiseless.get_optimization_stage_plugin() == 'ab_optimization'

    logical = provider.get_backend('EMU:40Q:LOGICAL_TARGET')
    assert logical.get_optimization_stage_plugin() == 'ab_optimization'

    logical_early = provider.get_backend('EMU:15Q:LOGICAL_EARLY')
    assert logical_early.get_optimization_stage_plugin() == 'ab_optimization'


def test_physical_backend_has_no_optimization_plugin():
    """Test that physical backends do not have an optimization plugin."""
    provider = AliceBobLocalProvider()

    physical = provider.get_backend('EMU:6Q:PHYSICAL_CATS')
    assert physical.get_optimization_stage_plugin() is None
