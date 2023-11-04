# pylint: disable=unused-argument

from qiskit_alice_bob_provider import AliceBobRemoteProvider


def test_list_backends(mocked_targets) -> None:
    provider = AliceBobRemoteProvider(api_key='foo')
    backends = provider.backends()
    for backend in backends:
        assert backend.name in str(backend)
