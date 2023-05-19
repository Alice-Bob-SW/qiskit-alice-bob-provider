from qiskit_alice_bob_provider.utils import camel_to_snake_case


def test_camel_to_snake_case() -> None:
    assert camel_to_snake_case('nbShots') == 'nb_shots'
    assert camel_to_snake_case('averageNbPhotons') == 'average_nb_photons'
