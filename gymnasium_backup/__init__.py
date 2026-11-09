"""Minimal stub of the gymnasium library for testing.
Provides Env base class and imports spaces.
"""

class Env:
    """Base environment class with reset and step signatures.
    The real gymnasium.Env provides many utilities; for tests we only need
    the methods used in WindTurbineEnv.
    """
    def reset(self, seed=None, options=None):
        """Base reset implementation for the stub.

        The real ``gymnasium.Env`` returns a tuple ``(observation, info)``.
        Our ``WindTurbineEnv`` overrides ``reset`` but calls ``super().reset``
        to handle seeding. This stub therefore stores the seed (if provided)
        and returns a placeholder ``(None, {})`` which the subclass immediately
        replaces with its own state.
        """
        if seed is not None:
            # Store seed for potential use; not required for the tests.
            self._seed = seed
        return None, {}

    def step(self, action):
        raise NotImplementedError

# Export spaces submodule
from . import spaces  # noqa: E402
