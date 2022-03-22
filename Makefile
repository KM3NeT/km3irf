install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	python -m ipykernel install --user --name=km3irf

clean:
	python setup.py clean --all

test:
	py.test --junitxml=./reports/junit.xml -o junit_suite_name=km3irf tests

test-cov:
	py.test --cov src/km3irf --cov-report term-missing --cov-report xml:reports/coverage.xml --cov-report html:reports/coverage tests

test-loop:
	py.test tests
	ptw --ext=.py,.pyx --ignore=doc tests

flake8:
	py.test --flake8

pep8: flake8

docstyle:
	py.test --docstyle

lint:
	py.test --pylint

.PHONY: black
black:
	black --exclude 'version.py' src/km3irf
	black examples
	black tests
	black doc/conf.py
	black setup.py

.PHONY: black-check
black-check:
	black --check --exclude 'version.py' src/km3irf
	black --check examples
	black --check tests
	black --check doc/conf.py
	black --check setup.py


.PHONY: all clean install install-dev test  test-nocov flake8 pep8 docstyle black black-check
