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
