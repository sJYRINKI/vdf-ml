"""Expose lightweight project-wide names without importing workflows.

Process-based dataset workers import this package before loading their
data-stage function. Keeping package initialization independent from the
representation, analysis, and model packages prevents those workers from
loading unrelated workflow modules while deserializing a task.
"""


RAW = "raw"

__all__ = ["RAW"]
