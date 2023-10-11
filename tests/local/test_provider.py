from qiskit_alice_bob_provider.local.provider import AliceBobLocalProvider


def test_get_backends() -> None:
    ab = AliceBobLocalProvider()
    ab.backends()
    assert (
        ab.get_backend('EMU:6Q:PHYSICAL_CATS').name == 'EMU:6Q:PHYSICAL_CATS'
    )
    backends = ab.backends('EMU:6Q:PHYSICAL_CATS')
    assert len(backends) == 1
    assert ab.get_backend('EMU:6Q:PHYSICAL_CATS').name == backends[0].name
