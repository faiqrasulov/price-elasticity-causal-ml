PYTHON ?= python
MEMCAP ?=

.PHONY: install data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Write the simulated main panel to data/ for inspection (no external data is used).
data:
	$(MEMCAP) $(PYTHON) -m elasticity.cli simulate --out data/simulated_panel.csv.gz

# Full reproduction: main analysis + Monte Carlo -> reports/metrics.json + reports/figures/*.png
run:
	$(MEMCAP) $(PYTHON) -m elasticity.cli run --out reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
