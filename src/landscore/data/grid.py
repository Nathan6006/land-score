"""The analysis grid: EPSG:5070, 30 m, aligned to the Annual NLCD pixel grid."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from affine import Affine

# Annual NLCD / NLCD CONUS grid origin (upper-left corner, EPSG:5070). Pixel edges fall on
# multiples of 30 m offset by these values. Verified against the source file at fetch time.
NLCD_ORIGIN_X = -2493045.0
NLCD_ORIGIN_Y = 3310005.0


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: tuple[float, ...]  # affine a, b, c, d, e, f
    width: int
    height: int

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform[:6])

    @property
    def res(self) -> float:
        return self.transform[0]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        a = self.affine
        return (a.c, a.f + a.e * self.height, a.c + a.a * self.width, a.f)

    def profile(self, dtype: str, count: int = 1, nodata=None) -> dict:
        return {
            "driver": "GTiff",
            "crs": self.crs,
            "transform": self.affine,
            "width": self.width,
            "height": self.height,
            "count": count,
            "dtype": dtype,
            "nodata": nodata,
            "compress": "deflate",
            "predictor": 2 if dtype.startswith(("int", "uint")) else 3,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Grid:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["crs"], tuple(d["transform"]), d["width"], d["height"])


def snap_grid(
    bounds: tuple[float, float, float, float],
    res: float = 30.0,
    origin: tuple[float, float] = (NLCD_ORIGIN_X, NLCD_ORIGIN_Y),
    crs: str = "EPSG:5070",
) -> Grid:
    """Smallest grid aligned to `origin` that covers `bounds` (xmin, ymin, xmax, ymax)."""
    xmin, ymin, xmax, ymax = bounds
    ox, oy = origin
    x0 = ox + math.floor((xmin - ox) / res) * res
    x1 = ox + math.ceil((xmax - ox) / res) * res
    y1 = oy - math.floor((oy - ymax) / res) * res
    y0 = oy - math.ceil((oy - ymin) / res) * res
    width = round((x1 - x0) / res)
    height = round((y1 - y0) / res)
    t = Affine(res, 0.0, x0, 0.0, -res, y1)
    return Grid(crs, tuple(t)[:6], width, height)
