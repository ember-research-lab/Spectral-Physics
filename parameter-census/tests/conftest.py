"""Pytest configuration for parameter-census tests.

Maps the hyphenated directory name to a Python-importable module.
"""

import importlib
import sys
from pathlib import Path

# Add the repo root so that `parameter-census/` can be found,
# then register the package under the underscore name.
_repo_root = Path(__file__).resolve().parents[2]
_pkg_dir = _repo_root / "parameter-census"

if "parameter_census" not in sys.modules:
    sys.path.insert(0, str(_repo_root))
    spec = importlib.util.spec_from_file_location(
        "parameter_census",
        str(_pkg_dir / "__init__.py"),
        submodule_search_locations=[str(_pkg_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["parameter_census"] = mod
    spec.loader.exec_module(mod)
