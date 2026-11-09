"""Stub for fastapi.responses.
Provides FileResponse used in backend routes.
"""

class FileResponse:
    def __init__(self, path, media_type=None, filename=None):
        self.path = path
        self.media_type = media_type
        self.filename = filename

    def __repr__(self):
        return f"FileResponse(path={self.path}, media_type={self.media_type}, filename={self.filename})"
