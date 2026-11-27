import sys
from pathlib import Path

PLATFORM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLATFORM))
import autoscaler._paths  # noqa: E402,F401  (adds <repo>/src)
