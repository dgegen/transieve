import importlib
from importlib.metadata import version, PackageNotFoundError
from typing import Any

from .lightcurve import LightCurve
from .simulation import SimulatedLightCurve

__all__ = [
    "__version__",
    "LightCurve",
    "SimulatedLightCurve",
]

_lazy_submodules = {
    "gp": "transieve.gp",
    "lightcurve": "transieve.lightcurve",
    "transit": "transieve.transit",
    "utils": "transieve.utils",
    "wavelet": "transieve.wavelet",
}


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"


def __getattr__(name: str) -> Any:
    if name in _lazy_submodules:
        module = importlib.import_module(_lazy_submodules[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_submodules.keys()))
