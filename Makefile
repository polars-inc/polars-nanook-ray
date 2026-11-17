SHELL=/bin/bash

PYTHON_VERSION?=3.12

LICENSE_PATH="$(PWD)/_etc/license.json"

.venv:
	@uv venv --python=$(PYTHON_VERSION) --managed-python
	@uv pip install --requirement=requirements-dev.txt
	@uv pip install --editable .

test: .venv
	@source .venv/bin/activate && \
	 LICENSE_PATH=$(LICENSE_PATH) \
	 pytest --verbose

clean:
	@rm -fr .mypy_cache .pytest_cache .ropeproject .ruff_cache uv.lock .venv
	@rm -fr $$(find . -name __pycache__)
