"""Minimal stub of the FastAPI library for testing.
Provides only the symbols used in the HMARL-DRT codebase.
"""

from typing import Callable, Any, Dict, List

class FastAPI:
    def __init__(self, title: str = "", description: str = "", version: str = ""):
        self.title = title
        self.description = description
        self.version = version
        self.routes = []
        self.middleware = []

    def add_middleware(self, middleware_class, **options):
        # Store middleware class and options for introspection (no-op).
        self.middleware.append((middleware_class, options))

    def include_router(self, router):
        # Simple inclusion: extend routes list.
        if hasattr(router, "routes"):
            self.routes.extend(router.routes)

    # Decorators for HTTP methods – used in route modules.
    def get(self, path: str, **kwargs):
        def decorator(func: Callable):
            self.routes.append({"method": "GET", "path": path, "func": func, "kwargs": kwargs})
            return func
        return decorator

    def post(self, path: str, **kwargs):
        def decorator(func: Callable):
            self.routes.append({"method": "POST", "path": path, "func": func, "kwargs": kwargs})
            return func
        return decorator

    def put(self, path: str, **kwargs):
        def decorator(func: Callable):
            self.routes.append({"method": "PUT", "path": path, "func": func, "kwargs": kwargs})
            return func
        return decorator

    def delete(self, path: str, **kwargs):
        def decorator(func: Callable):
            self.routes.append({"method": "DELETE", "path": path, "func": func, "kwargs": kwargs})
            return func
        return decorator

class HTTPException(Exception):
    def __init__(self, status_code: int, detail: Any = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")

# Export submodule for middleware
from . import middleware  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal APIRouter implementation (used by backend route modules)
# ---------------------------------------------------------------------------

class APIRouter:
    """Simple router that records route definitions.

    The real FastAPI APIRouter supports many features; for the test suite we
    only need to store the path, HTTP method, and the handler function.
    """

    def __init__(self, prefix: str = "", tags: List[str] | None = None):
        self.prefix = prefix
        self.tags = tags or []
        self.routes: List[Dict[str, Any]] = []

    def _add_route(self, method: str, path: str, **kwargs):
        def decorator(func: Callable):
            full_path = f"{self.prefix}{path}"
            self.routes.append({"method": method, "path": full_path, "func": func, "kwargs": kwargs})
            return func
        return decorator

    def get(self, path: str, **kwargs):
        return self._add_route("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self._add_route("POST", path, **kwargs)

    def put(self, path: str, **kwargs):
        return self._add_route("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self._add_route("DELETE", path, **kwargs)
