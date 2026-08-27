.PHONY: validate quick full refit test clean

validate:
	python scripts/00_validate.py --strict

quick:
	python scripts/reproduce.py --mode quick

full:
	python scripts/reproduce.py --mode full

refit:
	python scripts/01_refit_logistic.py

test:
	pytest

clean:
	python scripts/99_clean.py

