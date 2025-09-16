from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT_DIR / "data"
RAW_DIR   = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INDEX_DIR = DATA_DIR / "index"
PAIRS_CSV = DATA_DIR / "pairs.csv"

for d in (RAW_DIR, CLEAN_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# parâmetros
LBP_P = 16
LBP_R = 2
RESIZE_TARGET = 256  # menor lado após crop
