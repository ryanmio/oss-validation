import pkgutil, importlib, sys, unittest.mock as m, pathlib, pandas as pd
try:
    import geopandas as gpd
except ImportError:
    gpd = None

NULL = m.MagicMock()
PATCHES = [
    m.patch.object(pd.DataFrame,   "to_csv",  NULL),
    m.patch.object(pathlib.Path,   "write_text", NULL),
    m.patch.object(pathlib.Path,   "write_bytes", NULL),
]
if gpd:
    PATCHES.append(m.patch.object(gpd.GeoDataFrame, "to_file", NULL))
for p in PATCHES: p.start()

def sweep_pkg(pkg_name):
    for mod in pkgutil.walk_packages(pkg_name.__path__, pkg_name.__name__ + "."):
        try:
            importlib.import_module(mod.name)
        except Exception as e:
            print(f"\u274C {mod.name}: {e}", file=sys.stderr)

import oss_preprocessing, oss_validation, archive
for pkg in (oss_preprocessing, oss_validation, archive):
    sweep_pkg(pkg)

for p in PATCHES: p.stop() 