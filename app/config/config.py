"""
Backwards-compatibility module forwarding settings to app.config.settings.
"""
from app.config.settings import settings, Settings

__all__ = ["settings", "Settings"]
