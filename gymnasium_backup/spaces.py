"""Stub for gymnasium.spaces.
Provides Box class used in WindTurbineEnv.
"""

import numpy as np

class Box:
    def __init__(self, low, high, dtype=None):
        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high)
