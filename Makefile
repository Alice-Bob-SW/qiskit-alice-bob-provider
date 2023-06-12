VENV=venv
PYTHON=python
MODULES=qiskit_alice_bob_provider tests
BUILD_DIR=dist
BASE_URL=
API_KEY=
ifeq ($(OS),Windows_NT)
    ACTIVATE=$(VENV)/Scripts/activate
else
    ACTIVATE=$(VENV)/bin/activate
endif

#### Virtual environment

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	. $(ACTIVATE) && pip install -e .[dev]

clear:
	rm -rf $(VENV)

#### Formatting

_black:
	. $(ACTIVATE) && black $(MODULES)

_isort:
	. $(ACTIVATE) && isort $(MODULES)

_single_quotes:
	. $(ACTIVATE) && \
		pre-commit run --all-files double-quote-string-fixer

format: _black _isort _single_quotes

#### Linting

_check_black:
	. $(ACTIVATE) && black --check $(MODULES)

_check_isort:
	. $(ACTIVATE) && isort --check-only $(MODULES)

_flake8:
	. $(ACTIVATE) && flake8 $(MODULES)

_pylint:
	. $(ACTIVATE) && pylint $(MODULES)

_mypy:
	. $(ACTIVATE) && mypy $(MODULES)

lint: _check_black _check_isort _mypy _flake8 _pylint

#### Tests

unit-tests:
	. $(ACTIVATE) && pytest tests/


integration-tests:
	. $(ACTIVATE) && pytest --base-url=$(BASE_URL) \
	    --api-key=$(API_KEY) tests/

coverage:
	. $(ACTIVATE) \
		&& coverage run --omit='tests/*' -m pytest tests/ \
		&& coverage html \
		&& open htmlcov/index.html

test: lint unit-tests


#### Build

build: install
	rm -rf $(BUILD_DIR)
	. $(ACTIVATE) && python -m build -o $(BUILD_DIR)

test-publish: build
	. $(ACTIVATE) && python -m twine upload -r testpypi dist/*

publish: build
	. $(ACTIVATE) && python -m twine upload dist/*
