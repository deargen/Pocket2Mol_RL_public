from pathlib import Path
import os

__version__ = "0.1.0"

ROOT_DIR = Path(__file__).parent.parent


def replace_root_dir(path):
    # replace first "./" with ROOT_DIR
    if isinstance(path, Path):
        path = str(path)
    if path.startswith("./"):
        path = path.replace("./", "", 1)
        path = os.path.join(ROOT_DIR, path)
    return path
