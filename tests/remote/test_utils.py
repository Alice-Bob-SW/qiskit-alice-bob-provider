from qiskit_alice_bob_provider.remote.utils import (
    camel_to_snake_case,
    snake_to_camel_case,
)


def test_camel_to_snake_case() -> None:
    assert camel_to_snake_case('nbShots') == 'nb_shots'
    assert camel_to_snake_case('averageNbPhotons') == 'average_nb_photons'


def test_snake_to_camel_case() -> None:
    assert snake_to_camel_case('nb_shots') == 'nbShots'
    assert snake_to_camel_case('average_nb_photons') == 'averageNbPhotons'
