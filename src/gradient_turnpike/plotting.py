"""Shared, lightweight plotting conventions for paper and diagnostic figures."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save matched PNG and vector-PDF renderings with consistent layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
