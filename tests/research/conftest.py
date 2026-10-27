import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "src" / "research"
sys.path.insert(0, str(R))
