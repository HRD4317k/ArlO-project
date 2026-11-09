"""Stub for fastapi.middleware.cors.CORSMiddleware.
Only the constructor signature is needed for the backend import.
"""

class CORSMiddleware:
    def __init__(self, app, allow_origins=None, allow_credentials=False, allow_methods=None, allow_headers=None):
        # Store parameters for potential introspection; no functional behavior.
        self.app = app
        self.allow_origins = allow_origins or []
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or []
        self.allow_headers = allow_headers or []
