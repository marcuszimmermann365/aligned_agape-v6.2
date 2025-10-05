
from pathlib import Path
import re

def _tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zäöüß0-9\s_\-]", " ", text)
    return [t for t in text.split() if t]

class RAGStore:
    def __init__(self, base_path):
        self.base = Path(base_path)
    def fetch_passages(self, rel_path, k=1):
        file = (self.base / rel_path)
        if not file.exists():
            return []
        lines = [l.strip() for l in file.read_text(encoding='utf-8').splitlines() if l.strip()]
        out = []
        for i, l in enumerate(lines[:k]):
            out.append({"id": f"{file.name}:{i+1}", "text": l})
        return out
    def query(self, actor_spec):
        return self.fetch_passages(actor_spec.get("corpus",""), k=1)
