import pytest

from qiskit_alice_bob_provider.local.noise_model import NoiseModel, TimeModel


@pytest.fixture
def simple_noise_model():
    def noise_fn(_gate_params, _backend_params):
        return {
            'X': 0.1,
            'Y': 0.2,
            'Z': 0.3,
        }

    return NoiseModel(gate_name='rx', noise_fn=noise_fn)


@pytest.fixture
def params_noise_model():
    def noise_fn(gate_params, backend_params):
        return {
            'X': gate_params[0] * 0.1,
            'Y': backend_params['param1'] * 0.1,
            'Z': 0.1,
        }

    return NoiseModel(gate_name='cx', noise_fn=noise_fn)


@pytest.fixture
def simple_time_model():
    def time_fn(_gate_params, _backend_params):
        return 100.0

    return TimeModel(gate_name='rx', time_fn=time_fn)


@pytest.fixture
def params_time_model():
    def time_fn(gate_params, backend_params):
        return gate_params[0] * backend_params['clock_speed']

    return TimeModel(gate_name='cx', time_fn=time_fn)


def test_noise_model_returns_expected_values_for_simple_model():
    # Given
    def noise_fn(_gate_params, _backend_params):
        return {
            'X': 0.1,
            'Y': 0.2,
            'Z': 0.3,
        }

    model = NoiseModel(gate_name='rx', noise_fn=noise_fn)

    # When
    result = model.apply([], {})

    # Then
    assert result == {'X': 0.1, 'Y': 0.2, 'Z': 0.3}


def test_noise_model_uses_parameters_for_calculation():
    # Given
    def noise_fn(gate_params, backend_params):
        return {
            'X': gate_params[0] * 0.1,
            'Y': backend_params['param1'] * 0.1,
            'Z': 0.1,
        }

    model = NoiseModel(gate_name='cx', noise_fn=noise_fn)

    # When
    result = model.apply([0.5], {'param1': 0.3})

    # Then
    assert result == {'X': 0.05, 'Y': 0.03, 'Z': 0.1}


def test_noise_model_raises_error_for_probabilities_greater_than_one():
    # Given
    def invalid_noise_fn(_, __):
        return {'X': 0.5, 'Y': 0.6, 'Z': 0.2}

    invalid_model = NoiseModel(gate_name='invalid', noise_fn=invalid_noise_fn)

    # When / Then
    with pytest.raises(ValueError) as excinfo:
        invalid_model.apply([], {})
    assert 'Total noise probability must be in the range [0, 1]' in str(
        excinfo.value
    )


def test_noise_model_raises_error_for_negative_probabilities():
    # Given
    def invalid_noise_fn(_, __):
        return {'X': -0.1, 'Y': 0.1, 'Z': 0.1}

    invalid_model = NoiseModel(gate_name='invalid', noise_fn=invalid_noise_fn)

    # When / Then
    with pytest.raises(ValueError) as excinfo:
        invalid_model.apply([], {})
    assert 'Noise probabilities must be in the range [0, 1]' in str(
        excinfo.value
    )


def test_noise_model_caches_function_calls():
    # Given
    def noise_fn(gate_params, backend_params):
        return {
            'X': gate_params[0] * 0.1,
            'Y': backend_params['param1'] * 0.1,
            'Z': 0.1,
        }

    model = NoiseModel(gate_name='cx', noise_fn=noise_fn)
    gate_params = [0.5]
    backend_params = {'param1': 0.3}

    # When
    model.apply(gate_params, backend_params)
    cache_info_first = model.cache_info()
    model.apply(gate_params, backend_params)
    cache_info_second = model.cache_info()

    # Then
    assert cache_info_first.hits == 0
    assert cache_info_first.misses == 1
    assert cache_info_second.hits == 1
    assert cache_info_second.misses == 1


def test_noise_model_clears_cache():
    # Given
    def noise_fn(gate_params, backend_params):
        return {
            'X': gate_params[0] * 0.1,
            'Y': backend_params['param1'] * 0.1,
            'Z': 0.1,
        }

    model = NoiseModel(gate_name='cx', noise_fn=noise_fn)
    gate_params = [0.5]
    backend_params = {'param1': 0.3}
    model.apply(gate_params, backend_params)

    # When
    model.clear_cache()
    cache_info = model.cache_info()

    # Then
    assert cache_info.currsize == 0


def test_time_model_returns_expected_duration():
    # Given
    def time_fn(_gate_params, _backend_params):
        return 100.0

    model = TimeModel(gate_name='rx', time_fn=time_fn)

    # When
    result = model.apply([], {})

    # Then
    assert result == 100.0


def test_time_model_uses_parameters_for_calculation():
    # Given
    def time_fn(gate_params, backend_params):
        return gate_params[0] * backend_params['clock_speed']

    model = TimeModel(gate_name='cx', time_fn=time_fn)

    # When
    result = model.apply([0.5], {'clock_speed': 2.0})

    # Then
    assert result == 1.0


def test_time_model_raises_error_for_negative_duration():
    # Given
    def invalid_time_fn(_, __):
        return -10.0

    invalid_model = TimeModel(gate_name='invalid', time_fn=invalid_time_fn)

    # When / Then
    with pytest.raises(ValueError) as excinfo:
        invalid_model.apply([], {})
    assert 'Duration must be a non-negative number' in str(excinfo.value)


def test_time_model_caches_function_calls():
    # Given
    def time_fn(gate_params, backend_params):
        return gate_params[0] * backend_params['clock_speed']

    model = TimeModel(gate_name='cx', time_fn=time_fn)
    gate_params = [0.5]
    backend_params = {'clock_speed': 2.0}

    # When
    model.apply(gate_params, backend_params)
    cache_info_first = model.cache_info()
    model.apply(gate_params, backend_params)
    cache_info_second = model.cache_info()

    # Then
    assert cache_info_first.hits == 0
    assert cache_info_first.misses == 1
    assert cache_info_second.hits == 1
    assert cache_info_second.misses == 1


def test_time_model_clears_cache():
    # Given
    def time_fn(gate_params, backend_params):
        return gate_params[0] * backend_params['clock_speed']

    model = TimeModel(gate_name='cx', time_fn=time_fn)
    gate_params = [0.5]
    backend_params = {'clock_speed': 2.0}
    model.apply(gate_params, backend_params)

    # When
    model.clear_cache()
    cache_info = model.cache_info()

    # Then
    assert cache_info.currsize == 0
