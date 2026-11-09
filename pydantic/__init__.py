"""Minimal stub of the pydantic library for testing.
Only the BaseModel class and Field function used in the codebase are provided.
"""

from typing import Any, Dict


def Field(default: Any = None, **kwargs):
    """Placeholder for pydantic.Field – simply returns the default value.
    The real function provides metadata; for tests we only need the value.
    """
    return default


class BaseModel:
    """Very small subset of pydantic.BaseModel.
    It stores provided keyword arguments as attributes and implements ``dict``
    and ``json`` methods used by FastAPI responses.
    """

    def __init__(self, **data: Any):
        for key, value in data.items():
            setattr(self, key, value)

    def dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dict__ if not k.startswith('_')}

    def json(self) -> str:
        import json
        return json.dumps(self.dict())
