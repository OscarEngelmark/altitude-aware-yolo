"""Shared matplotlib style presets for report (PDF) and PPT (PNG) output."""

import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple

REPORT = "report"
PPT = "ppt"
STYLES: List[str] = [REPORT, PPT]

_RC: Dict[str, Dict[str, Any]] = {
    REPORT: {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
    },
    PPT: {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.8,
    },
}

# (total_width_inches, height_per_row_inches) — used when n_cols == 1
_SIZES: Dict[str, Tuple[float, float]] = {
    REPORT: (5.5, 2.8),
    PPT:    (7.0, 3.5),
}
# width per column for multi-column figures
_COL_WIDTH: Dict[str, float] = {
    REPORT: 2.8,
    PPT:    3.0,
}

_FMT: Dict[str, str] = {REPORT: "pdf", PPT: "png"}
_DPI: Dict[str, int] = {REPORT: 150, PPT: 600}


def apply_style(style: str) -> None:
    """Apply rcParams for the given style preset."""
    if style not in _RC:
        raise ValueError(f"Unknown style {style!r}; choose from {STYLES}")
    plt.rcParams.update(_RC[style])


def figsize(
    style: str, n_rows: int = 1, n_cols: int = 1
) -> Tuple[float, float]:
    """Return (width, height) in inches for the given style and grid shape."""
    _, h_per_row = _SIZES[style]
    w = _SIZES[style][0] if n_cols == 1 else _COL_WIDTH[style] * n_cols
    return (w, h_per_row * n_rows)


def output_fmt(style: str) -> str:
    """Return the file format string ('pdf' or 'png')."""
    return _FMT[style]


def save_dpi(style: str) -> int:
    """Return the DPI for savefig (relevant for PNG output)."""
    return _DPI[style]
