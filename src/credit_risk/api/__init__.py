"""FastAPI serving layer for the frozen Credit Risk Model V1."""

from .app import app, create_app

__all__ = ["app", "create_app"]
