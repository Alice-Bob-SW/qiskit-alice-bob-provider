from qiskit_alice_bob_provider import AliceBobProvider


def test_list_backends() -> None:
    provider = AliceBobProvider(api_key='foo')
    backends = provider.backends()
    for backend in backends:
        assert backend.name in str(backend)
