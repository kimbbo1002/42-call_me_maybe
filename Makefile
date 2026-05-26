PYTHON = uv run python3
MAIN = main.py
FILE = launcher.py
BENCH = launcher_bench.py
VENV = .venv
ARG ?=

install:
	uv sync

run:
	uv run $(FILE)

bench:
	uv run $(BENCH)

debug:
	$(PYTHON) -m pdb $(MAIN) $(ARG)

lint:
	flake8
	mypy . --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8
	mypy . --explicit-package-bases --strict

clean:
	rm -rf `find . -type d -name "__pycache__"`
	rm -rf .mypy_cache

fclean: clean
	rm -rf uv.lock
	rm -rf .venv

.PHONY: install run pick debug lint lint-strict clean fclean