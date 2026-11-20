from __future__ import annotations

from .api.routes import app, create_app

__all__ = ["app", "create_app"]

#imports all routes from the api.routes module and makes them available for fastapi