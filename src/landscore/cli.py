"""Command line entry point: `uv run landscore <command>`."""

from __future__ import annotations

import argparse
import importlib
import sys

# command -> (module, function, help). Modules are imported lazily so `--help` is fast
# and a missing optional piece does not break unrelated commands.
COMMANDS = {
    "fetch": ("landscore.data.fetch", "main", "Download and clip input layers for counties"),
    "quicklook": ("landscore.data.quicklook", "main", "Save quick-look PNGs of fetched layers"),
    "labels": ("landscore.labels.build", "main", "Build abandonment labels for counties"),
    "features": ("landscore.features.table", "main", "Build the sampled training table"),
    "train": ("landscore.models.train", "main", "Train baselines and models, write results"),
    "report": ("landscore.report.plot_report", "main", "Score a GeoJSON plot and write a report"),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"} or argv[0] not in COMMANDS:
        p = argparse.ArgumentParser(prog="landscore")
        sub = p.add_subparsers(dest="command", required=True)
        for name, (_, _, help_) in COMMANDS.items():
            sub.add_parser(name, help=help_)
        p.parse_args(argv or ["--help"])
        return 2
    module, func, _ = COMMANDS[argv[0]]
    return int(getattr(importlib.import_module(module), func)(argv[1:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
