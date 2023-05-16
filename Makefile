VENV=venv
PYTHON=python
MODULES=qiskit_alice_bob_provider tests
BASE_URL=
API_KEY=

#### Virtual environment

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	ls
	ls venv/
	ls venv/*
	$(VENV)/bin/pip install -e .[dev]

clear:
	rm -rf $(VENV)

#### Formatting

_black:
	. $(VENV)/bin/activate && black $(MODULES)

_isort:
	. $(VENV)/bin/activate && isort $(MODULES)

_single_quotes:
	. $(VENV)/bin/activate && \
		pre-commit run --all-files double-quote-string-fixer

format: _black _isort _single_quotes

#### Linting

_check_black:
	. $(VENV)/bin/activate && black --check $(MODULES)

_check_isort:
	. $(VENV)/bin/activate && isort --check-only $(MODULES)

_flake8:
	. $(VENV)/bin/activate && flake8 $(MODULES)

_pylint:
	. $(VENV)/bin/activate && pylint $(MODULES)

_mypy:
	. $(VENV)/bin/activate && mypy $(MODULES)

lint: _check_black _check_isort _mypy _flake8 _pylint

#### Tests

unit-tests:
	. $(VENV)/bin/activate && pytest tests/


integration-tests:
	. $(VENV)/bin/activate && pytest --base-url=$(BASE_URL) \
	    --api-key=$(API_KEY) tests/

coverage:
	. $(VENV)/bin/activate \
		&& coverage run --omit='tests/*' -m pytest tests/ \
		&& coverage html \
		&& open htmlcov/index.html

test: lint unit-tests
