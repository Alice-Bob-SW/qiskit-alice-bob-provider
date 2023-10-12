from typing import Dict

import pytest

from qiskit_alice_bob_provider.remote.qir_to_qiskit import (
    ab_target_to_qiskit_target,
)


def test_single_cat(single_cat_target: Dict) -> None:
    ab_target_to_qiskit_target(single_cat_target)


def test_all_instructions(all_instructions_target: Dict) -> None:
    ab_target_to_qiskit_target(all_instructions_target)


def test_unknown_qis_instruction(all_instructions_target: Dict) -> None:
    all_instructions_target['instructions'].append(
        {'signature': '__quantum__qis__foo__body:void (%Qubit*)'}
    )
    with pytest.warns():
        ab_target_to_qiskit_target(all_instructions_target)


def test_non_qis_instruction(all_instructions_target: Dict) -> None:
    all_instructions_target['instructions'].append(
        {'signature': 'foo:void ()'}
    )
    with pytest.warns():
        ab_target_to_qiskit_target(all_instructions_target)


def test_signature_without_arguments(all_instructions_target: Dict) -> None:
    all_instructions_target['instructions'].append({'signature': 'foo:void'})
    with pytest.warns():
        ab_target_to_qiskit_target(all_instructions_target)
