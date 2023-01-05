version := `python3 -c "from src.ntcad.__about__ import __version__; print(__version__)"`

# Cleans the repo.
clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo|build|generated$)" | xargs rm -rf
	@rm -rf src/*.egg-info/ build/ dist/ .tox/

# Applies formatting to all files.
format:
	isort .
	black .
	blacken-docs

# Lints all files.
lint:
	black --check .
	flake8 .

# Builds the documentation.
doc: lint
	cd doc && make html
