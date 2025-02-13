VENV=venv
PIP?=pip

MODULES=qiskit_alice_bob_provider tests
BUILD_DIR=dist
BASE_URL=
API_KEY=
ifeq ($(OS),Windows_NT)
    ACTIVATE=$(VENV)/Scripts/activate
    PYTHON3_PATH=$(cmd which python3 2> nul)
else
    ACTIVATE=$(VENV)/bin/activate
    PYTHON3_PATH=$(shell which python3 2> /dev/null)
endif

ifeq ($(PYTHON3_PATH),)
    PYTHON=python
else
    PYTHON=python3
endif

#### Virtual environment

$(VENV):
	$(PYTHON) -m venv $(VENV)

install: $(VENV)
	. $(ACTIVATE) && $(PIP) install -e .[dev]

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

### Precommit
precommit-hooks:
	. $(ACTIVATE) && pre-commit install \
		&& pre-commit install --hook-type commit-msg

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
