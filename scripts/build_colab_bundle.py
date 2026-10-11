"""Archive committed source only; never includes caches, credentials or results outputs."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
out = root / "colab_outputs" / "source"
out.mkdir(parents=True, exist_ok=True)
bundle = out / f"mmrag-{sha}.zip"
subprocess.run(["git", "archive", "--format=zip", "-o", str(bundle), sha], cwd=root, check=True)
manifest = {
    "source_commit": sha,
    "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
    "bundle": bundle.name,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
