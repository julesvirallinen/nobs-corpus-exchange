"""SPINE mechanism package.

Receives ONLY Text. Has no knowledge of the other CSV columns — the protected
field name does not appear anywhere in this package, and a test enforces it.

Public API:
    privatize(text, config) -> (new_text, token_log)
    identity(text)          -> (text, token_log)
    load_config(path)       -> Config
"""

from .config import Config, config_from_dict, load_config
from .spine import identity, privatize

__all__ = ["privatize", "identity", "load_config", "Config", "config_from_dict"]
