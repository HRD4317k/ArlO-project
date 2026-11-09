"""Minimal stub of plotly.graph_objects for testing.
Provides only the symbols used in the dashboard components.
"""

class Figure:
    def __init__(self, data=None, **kwargs):
        """Stub Figure constructor.
        Accepts an optional ``data`` argument (a trace dict or list of traces)
        used by the radar chart implementation. Any additional kwargs are
        ignored as they are not needed for the tests.
        """
        self.traces = []
        if data is not None:
            # Normalise to list of (trace, secondary_y) tuples.
            if isinstance(data, list):
                for tr in data:
                    self.traces.append((tr, False))
            else:
                self.traces.append((data, False))
        self.layout = {}
        self._yaxes = {}

    def add_trace(self, trace, secondary_y=False):
        # Store trace with optional secondary flag
        self.traces.append((trace, secondary_y))
        return self

    def update_layout(self, **kwargs):
        self.layout.update(kwargs)
        return self

    def update_yaxes(self, **kwargs):
        # Store yaxis settings; ignore secondary handling for stub
        self._yaxes.update(kwargs)
        return self

    def add_vrect(self, **kwargs):
        # Store vertical rectangle annotation (no-op for stub)
        self.layout.setdefault('vrects', []).append(kwargs)
        return self

# Simple trace factories returning dicts
def Scatter(*, x=None, y=None, mode=None, name=None, marker=None, line=None, **kwargs):
    """Stub for plotly.graph_objects.Scatter.
    Accepts any additional keyword arguments (e.g., ``stackgroup``) and stores
    them in the returned dict for compatibility. Unused arguments are ignored
    by the test suite.
    """
    data = {
        'type': 'scatter',
        'x': x,
        'y': y,
        'mode': mode,
        'name': name,
        'marker': marker or {},
        'line': line or {},
    }
    # Merge any extra kwargs without validation.
    data.update(kwargs)
    return data

def Bar(*, x=None, y=None, name=None, marker_color=None, opacity=None):
    return {
        'type': 'bar',
        'x': x,
        'y': y,
        'name': name,
        'marker': {'color': marker_color},
        'opacity': opacity,
    }

def Scatterpolar(*, r=None, theta=None, fill=None, name=None, **kwargs):
    """Stub for plotly.graph_objects.Scatterpolar used in radar charts.
    Accepts typical arguments and returns a dict representation.
    """
    data = {
        'type': 'scatterpolar',
        'r': r,
        'theta': theta,
        'fill': fill,
        'name': name,
    }
    data.update(kwargs)
    return data
