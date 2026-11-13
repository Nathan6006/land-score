# Convenience wrappers for macOS/Linux. On Windows run the `uv run ...` commands directly.
COUNTY ?= Geauga

.PHONY: setup test lint format fetch labels features train report all
setup:
	uv sync --locked
test:
	uv run pytest
lint:
	uv run ruff check . && uv run ruff format --check .
format:
	uv run ruff check --fix . && uv run ruff format .
fetch:
	uv run landscore fetch --county $(COUNTY)
labels:
	uv run landscore labels --county $(COUNTY)
features:
	uv run landscore features --county $(COUNTY)
train:
	uv run landscore train
report:
	uv run landscore report examples/plot_a.geojson
all: fetch labels features train
