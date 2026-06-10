PYTHON = uv run python3
MAIN = main.py
FUNC = data/input/functions_definition.json
INPUT = data/input/function_calling_tests.json
OUTPUT = data/output/function_calls.json

install:
	uv sync

run:
	uv run python -m src \
	--function_defs $(FUNC) \
	--input $(INPUT) \
	--output $(OUTPUT)

debug:
	$(PYTHON) -m pdb $(MAIN) $(ARG)

lint:
	flake8 src/
	mypy src/ --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=silent

lint-strict:
	flake8 src/
	mypy src/ --explicit-package-bases --strict --follow-imports=silent

clean:
	rm -rf `find . -type d -name "__pycache__"`
	rm -rf .mypy_cache

fclean: clean
	rm -rf uv.lock
	rm -rf .venv
	rm -rf data/output/

.PHONY: install run debug lint lint-strict clean fclean