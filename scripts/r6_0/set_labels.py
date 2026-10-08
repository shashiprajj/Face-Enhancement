"""R6-0 — record visual labels: python scripts/r6_0/set_labels.py INDEX_JSON LABEL "i,j,k" [note]
Labels: GOOD | BAD_CLEAR | BAD_FAINT | BAD_DESTRUCTIVE | UNCERTAIN. Stored in data/manifests/phase5_rapid/r6_0/labels_r6.json
keyed by case id, with the review sheet and note. Each label is a statement about the RETURNED output v0."""
import json, sys
from pathlib import Path
P = Path(__file__).resolve().parents[2] / "data/manifests/phase5_rapid/r6_0/labels_r6.json"
idx = json.loads(Path(sys.argv[1]).read_text())
lab = json.loads(P.read_text()) if P.exists() else {}
for i in sys.argv[3].split(","):
    if i.strip():
        lab[idx[i.strip()]] = {"label": sys.argv[2], "sheet": Path(sys.argv[1]).stem, "note": sys.argv[4] if len(sys.argv) > 4 else ""}
P.write_text(json.dumps(dict(sorted(lab.items())), indent=1))
print(len(lab))
