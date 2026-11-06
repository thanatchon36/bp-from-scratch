.PHONY: all test bench clean results figures paper arxiv docs

## all: tests -> figures and tables (from the committed results) -> paper (PDF, web reader, Markdown) -> docs site
all: test paper docs

test:
	python -m pytest -q

bench:
	python benchmarks/bench_gates.py --max-n 18 --layers 10

## results: re-run every experiment (~1 CPU hour; resumable, cached in results/)
results:
	BUDGET=100000 bash experiments/run_all.sh

figures:
	python analysis/make_figures.py && python analysis/make_tables.py

## paper: paper/paper.pdf (Tectonic), paper/paper.md and docs/paper/ (pandoc)
paper: figures
	bash paper/build.sh

## arxiv: dist/arxiv/bp-from-scratch-arxiv.tar.gz, test-compiled with pdfLaTeX when Docker is available
arxiv:
	bash paper/arxiv.sh

## docs: build the MkDocs site (site/) that GitHub Pages serves
docs:
	mkdocs build --strict

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + ; rm -rf .pytest_cache build dist site *.egg-info
