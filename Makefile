#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = nyc-taxi-mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 nyc_taxi_mlops
	isort --check --diff --profile black nyc_taxi_mlops
	black --check --config pyproject.toml nyc_taxi_mlops

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml nyc_taxi_mlops






#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) nyc_taxi_mlops/data/make_dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
