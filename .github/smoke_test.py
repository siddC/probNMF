"""Smoke test of an installed probnmf (CI runs it on the built wheel, outside the checkout).

Checks that probnmf is imported from site-packages and that every module in the package
imports, so a module left out of the wheel fails here.
"""

import importlib
import pathlib
import pkgutil

import probnmf

path = pathlib.Path(probnmf.__file__)
assert "site-packages" in path.parts, f"probnmf imported from {path}, not from the install"
assert (path.parent / "py.typed").exists(), "py.typed is missing from the wheel"
for mod in pkgutil.walk_packages(probnmf.__path__, "probnmf."):
    importlib.import_module(mod.name)
print(f"probnmf {probnmf.__version__} from {path.parent}: every module imports")
