.PHONY: test bench clean results figures paper

test:
	python -m pytest -q

bench:
	python benchmarks/bench_gates.py --max-n 18 --layers 10

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + ; rm -rf .pytest_cache build dist *.egg-info

results:
	BUDGET=100000 bash experiments/run_all.sh

figures:
	python analysis/make_figures.py && python analysis/make_tables.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && cp main.pdf paper.pdf
