from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "geomed/src"))
sys.path.insert(0, str(ROOT / "geomed/src/experiments"))
