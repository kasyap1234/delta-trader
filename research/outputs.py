from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp


def new_run_directory(name: str) -> Path:
    """Keep each run's CSVs together without overwriting earlier results."""
    root = Path("research/output")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(mkdtemp(prefix=f"{name}-{stamp}-", dir=root))
