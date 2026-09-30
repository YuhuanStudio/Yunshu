import subprocess
import sys
from pathlib import Path

w = Path(sys.argv[1])
subprocess.run([sys.executable, str(w / "report.py"), str(w / "data"), str(w / "report.json")], check=True)
