# Setup (about ten minutes)

You need: git, the GitHub CLI (`gh`) logged in, and **uv**. uv installs the right Python and
every package (including the geospatial libraries) from one lockfile, so you do not need
Homebrew, conda, or apt for this project.

## 1. Install uv (once per machine)

**macOS / Linux**
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then open a new terminal. Check with `uv --version`.

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you prefer not to pipe a script into a shell, `pip install uv` also works.

## 2. Clone and install

```sh
gh repo clone Nathan6006/land-score
cd land-score
uv sync --locked
cp .env.example .env        # Windows: copy .env.example .env
```

`uv sync` creates `.venv/` in the repo (about 1 GB). Edit `.env` if you want data stored on a
different drive (`LANDSCORE_DATA_DIR`).

## 3. Run the tests

```sh
uv run pytest
```

All tests should pass offline. Lint with `uv run ruff check .`.

## 4. Fetch pilot data (Geauga County)

```sh
uv run landscore fetch --county Geauga
uv run landscore quicklook --county Geauga
```

Expected download: see `docs/STATUS.md` for the current size estimate (pilot is well under
2 GB). Everything comes from public sources that need no account. Each file is recorded in
`data/manifest.json` with its URL, date, and SHA-256 checksum, so both machines get identical
inputs. Quick-look maps go to `outputs/quicklook/`.

## 5. Run the pipeline

```sh
uv run landscore labels --county Geauga
uv run landscore features --county Geauga
uv run landscore train
uv run landscore report examples/plot_a.geojson
```

On macOS/Linux, `make all COUNTY=Geauga` runs fetch → labels → features → train.

## Working together

- Read `docs/STATUS.md` before you start; update it before you stop.
- Pick an issue from the GitHub board, make a branch (`git switch -c 12-soil-features`),
  open a pull request, and let CI pass before merging. Never push to `main` directly.
- Log scope or design changes in `docs/DECISIONS.md`.
- Claude Code reads `CLAUDE.md` automatically, so both sessions share the same rules.

## Troubleshooting

- **`uv sync` fails on a geospatial package**: run `uv sync --locked -v` and paste the error
  into an issue. The lockfile pins wheels that bundle GDAL/PROJ for macOS (Intel and Apple
  Silicon), Windows, and Linux, so nothing should compile.
- **Out of disk**: set `LANDSCORE_DATA_DIR` in `.env` to an external drive.
- **Network timeouts during fetch**: rerun the same command; finished files are skipped
  (checked against the manifest).
