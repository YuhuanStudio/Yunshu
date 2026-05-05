"""Yunshu L2 Control Plane — Admin and monitoring API.

FastAPI-based control plane for:
- Model lifecycle management (CRUD, load/unload, priorities)
- System monitoring (GPU, memory, request stats)
- Authentication and RBAC
- Configuration management
"""

from .main import create_admin_app, app

__all__ = ["app", "create_admin_app"]
