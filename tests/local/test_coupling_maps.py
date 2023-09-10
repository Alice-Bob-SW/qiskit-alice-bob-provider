from qiskit_alice_bob_provider.local.coupling_maps import (
    bidirect_map,
    circular_map,
    rectangular_map,
)


def test_bidirect_map() -> None:
    map = bidirect_map([(0, 1), (3, 4)])
    assert (0, 1) in map
    assert (1, 0) in map
    assert (3, 4) in map
    assert (4, 3) in map


def test_circular_map() -> None:
    map = circular_map(3)
    assert len(map) == 6
    assert (0, 1) in map
    assert (1, 0) in map
    assert (1, 2) in map
    assert (2, 1) in map
    assert (2, 0) in map
    assert (0, 2) in map


def test_rectangular_map() -> None:
    map = rectangular_map(2, 3)
    assert len(map) == (1 * 3 + 2 * 2) * 2
