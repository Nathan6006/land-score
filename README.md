# Land Score

Land Score rates a plot of land in Northeast Ohio for how readily trees should establish
there. It is built for Roots of Tomorrow, a student-run nonprofit that plants native trees
on private land that landowners offer. Give it a plot boundary and it returns:

1. **Establishment score**: 0–100 for each 30 m pixel, a plot summary, and a map. This comes
   from a trained model.
2. **Species fit**: a short list of native trees suited to the plot's soil and drainage. This
   is a lookup table, not a model.
3. **Planning flags**: wet soil, heavy competing vegetation, likely deer pressure, steep slope.
   These are simple rules.

> **Site potential, not a survival promise.** The score describes how readily trees came back
> on their own on similar former farmland nearby. It is not a promise that planted trees will
> survive. Survival depends heavily on care after planting, especially weed control and deer
> protection.

## How it works (short version)

No public dataset links planted-tree outcomes to locations in Ohio. So the model learns from a
stand-in: fields that stopped being farmed between 1990 and 2005 (USGS Annual NLCD), and how
much tree canopy grew there on its own 15 years later (USFS Tree Canopy Cover). Features are
limited to things known before regrowth began: soil, terrain, climate, and how much forest was
nearby at the start. The model is validated on whole counties it never saw during training.
Full details: [`docs/METHODS.md`](docs/METHODS.md). Current results:
[`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

See [`docs/SETUP.md`](docs/SETUP.md). In short:

```sh
uv sync --locked
uv run pytest
uv run landscore fetch --county Geauga
```

## Project docs

- [`docs/STATUS.md`](docs/STATUS.md): handoff log, read first
- [`docs/DECISIONS.md`](docs/DECISIONS.md): every scope and design change, with reasons
- [`docs/METHODS.md`](docs/METHODS.md): data, labels, features, models, validation, limits
- [`CLAUDE.md`](CLAUDE.md): rules and conventions shared by our Claude Code sessions
