"""Predeclared optimizer learning-rate schedules for the frozen experiments."""


def staged_learning_rate(update: int) -> float:
    """Return 1e-3 to 1500, 3e-4 to 3000, and 1e-4 thereafter."""
    if update < 1:
        raise ValueError("Optimizer updates are one-indexed")
    return 1e-3 if update <= 1500 else 3e-4 if update <= 3000 else 1e-4
