"""Tools for a reproducible study of arbitrary binary choices in LLMs."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import app

    app()
