import re
from unittest.mock import Mock

import numpy as np
import pytest

from qiskit_alice_bob_provider import AliceBobLocalProvider
from qiskit_alice_bob_provider.processor.custom_cat import CustomCat


def test_create_custom_logical_cat_with_valid_parameters(
    default_custom_emulator_parameters,
):
    # When
    processor = CustomCat(
        **default_custom_emulator_parameters,
        validate_parameters=lambda params: params['n_qubits'] > 0,
        name='TestProcessor',
    )

    # Then
    assert processor.name == 'TestProcessor'
    assert processor.n_qubits == 40
    assert processor.clock_cycle == 1e-9
    assert 'kappa_1' in processor.backend_parameters
    assert 'kappa_2' in processor.backend_parameters
    assert 'average_nb_photons' in processor.backend_parameters
    assert 'distance' in processor.backend_parameters


def test_create_local_backend_with_custom_processor(
    default_custom_emulator_parameters,
):
    # When
    backend = AliceBobLocalProvider().build_custom_backend(
        **default_custom_emulator_parameters, name='EMU:CUSTOM_LOGICAL'
    )

    # Then
    assert backend is not None
    assert backend.name == 'EMU:CUSTOM_LOGICAL'
    # pylint: disable=protected-access
    assert backend._processor.n_qubits == 40
    # pylint: disable=protected-access
    assert backend._processor.clock_cycle == 1e-9
    assert (
        # pylint: disable=protected-access
        backend._processor.backend_parameters
        == default_custom_emulator_parameters['backend_parameters']
    )


def test_missing_required_parameters(default_custom_emulator_parameters):
    # Given
    params = default_custom_emulator_parameters
    params['backend_parameters'].pop('n_qubits')

    # When/Then
    with pytest.raises(
        ValueError,
        match="Backend parameters must include 'n_qubits'",
    ):
        CustomCat(**params)


def test_missing_noise_model_for_gate():
    # Given
    params = {'n_qubits': 2, 'clock_cycle': 1.0}
    noise_models = {'x': lambda a, b: {'X': 0, 'Y': 0, 'Z': 0}}
    time_models = {'x': lambda a, b: 0, 'z': lambda a, b: 0}

    processor = CustomCat(
        backend_parameters=params,
        noise_models=noise_models,
        time_models=time_models,
    )

    # When/Then
    with pytest.raises(ValueError, match="No noise model found for gate 'z'"):
        processor.apply_instruction('z', (0,), [])


def test_default_models_used_for_1q_gates(default_custom_emulator_parameters):
    # Given
    params = default_custom_emulator_parameters
    another_noise_fn = Mock(return_value={'X': 0.01, 'Y': 0.01, 'Z': 0.01})
    noise_fn = Mock(return_value={'X': 0.02, 'Y': 0.02, 'Z': 0.02})
    another_time_fn = Mock(return_value=0)
    time_fn = Mock(return_value=0)
    params['noise_models'] = {'x': another_noise_fn, 'z': another_noise_fn}
    params['time_models'] = {'x': another_time_fn, 'z': another_time_fn}
    params['default_1q_noise_model'] = noise_fn
    params['default_1q_time_model'] = time_fn
    processor = CustomCat(
        **params,
    )

    # When / Then
    processor.apply_instruction('h', (0,), [])
    noise_fn.assert_called_once_with([], processor.backend_parameters)
    another_noise_fn.assert_not_called()
    time_fn.assert_called_once_with([], processor.backend_parameters)
    another_time_fn.assert_not_called()


def test_lambda_with_no_parameters(default_custom_emulator_parameters):
    # Given
    params = default_custom_emulator_parameters
    params['noise_models'] = {'x': lambda: {'X': 0.01, 'Y': 0.01, 'Z': 0.01}}
    params['time_models'] = {'x': lambda: 0}
    processor = CustomCat(**params)

    # When / Then
    processor.apply_instruction('x', (0,), [])


def test_supported_gates_list(default_custom_emulator_parameters):
    # Given
    processor = CustomCat(**default_custom_emulator_parameters)

    # When
    instructions = list(processor.all_instructions())
    instruction_names = [inst.name for inst in instructions]

    # Then
    expected_instructions = [
        'delay',
        'x',
        'z',
        'p0',
        'p1',
        'p+',
        'p-',
        'mx',
        'mz',
        't',
        'tdg',
        'h',
        's',
        'sdg',
        'cx',
    ]
    assert all(instr in instruction_names for instr in expected_instructions)


def test_unsupported_gate_raises_error(default_custom_emulator_parameters):
    # Given
    processor = CustomCat(**default_custom_emulator_parameters)

    # When/Then
    with pytest.raises(
        ValueError,
        match="Instruction 'rz' is not supported by CustomCat",
    ):
        processor.apply_instruction('rz', (0,), [np.pi / 4])


def test_gate_duration_from_time_model(default_custom_emulator_parameters):
    # Given
    time_models = {
        'x': lambda params, backend_params: backend_params['clock_cycle'] * 2,
        'cx': lambda params, backend_params: backend_params['clock_cycle'] * 5,
    }
    default_custom_emulator_parameters['time_models'] = time_models
    default_custom_emulator_parameters['noise_models'] = {
        'x': lambda params, backend_params: {'X': 0.01, 'Y': 0.01, 'Z': 0.01},
        'cx': lambda params, backend_params: {'X': 0.02, 'Y': 0.02, 'Z': 0.02},
    }

    processor = CustomCat(**default_custom_emulator_parameters)

    # When
    x_instruction = processor.apply_instruction('x', (0,), [])
    cx_instruction = processor.apply_instruction('cx', (0, 1), [])

    # Then
    assert (
        x_instruction.duration
        == default_custom_emulator_parameters['backend_parameters'][
            'clock_cycle'
        ]
        * 2
    )
    assert (
        cx_instruction.duration
        == default_custom_emulator_parameters['backend_parameters'][
            'clock_cycle'
        ]
        * 5
    )


def test_invalid_noise_model_parameters(default_custom_emulator_parameters):
    # Given
    def invalid_noise_function(_params, _backend_params):
        return {'X': 1.2, 'Y': -0.1, 'Z': 0.5}

    default_custom_emulator_parameters[
        'default_1q_noise_model'
    ] = invalid_noise_function

    processor = CustomCat(**default_custom_emulator_parameters)

    # When/Then
    with pytest.raises(
        ValueError,
        match=re.escape('Noise probabilities must be in the range [0, 1].'),
    ):
        processor.apply_instruction('x', (0,), [])


def test_invalid_time_model_parameters(default_custom_emulator_parameters):
    # Given
    def invalid_time_function(_params, _backend_params):
        return -1

    default_custom_emulator_parameters[
        'default_1q_time_model'
    ] = invalid_time_function

    processor = CustomCat(**default_custom_emulator_parameters)

    # When/Then
    with pytest.raises(
        ValueError, match=re.escape('Duration must be a non-negative number.')
    ):
        processor.apply_instruction('x', (0,), [])


def test_can_validate_custom_parameters(default_custom_emulator_parameters):
    # Given
    def custom_validation(params):
        return params['kappa_1'] < 10

    with pytest.raises(
        ValueError,
        match=re.escape('Invalid parameters provided for CustomCat.'),
    ):
        CustomCat(
            **default_custom_emulator_parameters,
            validate_parameters=custom_validation,
        )
