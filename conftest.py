import sys
from pathlib import Path

# Add the project root to sys.path so `import src...` works in tests.[web:186]
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

