"""OSS Validation package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("oss-validation")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from . import config, parsing  # re-export for convenience
try:
    from . import export_gaps  # re-export new CLI helper
    __all__ = ["config", "export_gaps", "__version__"]
except ModuleNotFoundError:
    # Optional dependencies (network_adjustment) missing in current branch
    __all__ = ["config", "__version__"] 