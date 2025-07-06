"""OSS Validation package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("oss-validation")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from . import config, parsing  # re-export for convenience

__all__ = ["config", "__version__"] 