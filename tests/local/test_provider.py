import pytest

from qiskit_alice_bob_provider.local.backend import ProcessorSimulator
from qiskit_alice_bob_provider.local.provider import AliceBobLocalProvider
from qiskit_alice_bob_provider.processor.physical_cat import (
    PhysicalCatProcessor,
)


def test_get_backends() -> None:
    ab = AliceBobLocalProvider()
    ab.backends()
    assert (
        ab.get_backend('EMU:6Q:PHYSICAL_CATS').name == 'EMU:6Q:PHYSICAL_CATS'
    )
    backends = ab.backends('EMU:6Q:PHYSICAL_CATS')
    assert len(backends) == 1
    assert ab.get_backend('EMU:6Q:PHYSICAL_CATS').name == backends[0].name


# pylint: disable=protected-access
def test_get_backend_change_nbar() -> None:
    ab = AliceBobLocalProvider()
    backend = ab.get_backend('EMU:6Q:PHYSICAL_CATS', average_nb_photons=9)
    assert isinstance(backend, ProcessorSimulator)
    proc = backend.target.durations()._proc
    assert isinstance(proc, PhysicalCatProcessor)
    assert proc._average_nb_photons == pytest.approx(9)


def test_get_multiple_backends_with_options() -> None:
    """
    Test that getting multiple backends with different options does not affect
    the default Processor options (but backend.options will remain the same
    anyway for AliceBobLocalProvider).
    """
    provider = AliceBobLocalProvider()
    default_backend = provider.get_backend('EMU:6Q:PHYSICAL_CATS')
    proc1 = default_backend.target.durations()._proc
    _ = provider.get_backend('EMU:6Q:PHYSICAL_CATS', average_nb_photons=6)
    backend = provider.get_backend('EMU:6Q:PHYSICAL_CATS')
    proc2 = backend.target.durations()._proc

    # Ensure that the backend objects are different instances
    assert default_backend is not backend
    # Ensure that the Processor objects are different instances
    assert proc1 is not proc2
    # Ensure that the default Processor option didn't change
    assert proc1._average_nb_photons == proc2._average_nb_photons
