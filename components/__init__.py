"""Top‑level wrapper package for dashboard components.

The test suite expects to import ``components.api_client`` and ``components.charts``
as top‑level modules. The actual implementation lives under
``hmarl_drt/dashboard/components``. These thin wrappers re‑export the real
modules so that ``import components.api_client`` works without modifying the
original source tree.
"""

# Re‑export the actual component modules from their real location.
from importlib import import_module

# Dynamically import the real modules and expose them under the ``components``
# namespace. This avoids duplication and keeps a single source of truth.
api_client = import_module("hmarl_drt.dashboard.components.api_client")
charts = import_module("hmarl_drt.dashboard.components.charts")

__all__ = ["api_client", "charts"]
