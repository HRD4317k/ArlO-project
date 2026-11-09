"""Stub for plotly.express used in tests.
Only the functions referenced in the code are provided as simple wrappers.
"""

def scatter(*args, **kwargs):
    # Return a simple dict representing a figure
    return {'type': 'scatter', 'args': args, 'kwargs': kwargs}
