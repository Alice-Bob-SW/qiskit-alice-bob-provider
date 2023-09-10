from qiskit_alice_bob_provider.local.provider import AliceBobLocalProvider


def test_get_backends() -> None:
    ab = AliceBobLocalProvider()
    ab.backends()
    assert ab.get_backend('PHYSICAL_CATS_6').name == 'PHYSICAL_CATS_6'
    backends = ab.backends('PHYSICAL_CATS_6')
    assert len(backends) == 1
    assert ab.get_backend('PHYSICAL_CATS_6').name == backends[0].name
