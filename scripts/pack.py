import sys
from pathlib import Path

file = Path(sys.argv[1])
lines = file.read_text().splitlines()
packs = [lines[i : i + 25] for i in range(0, len(lines), 25)]

for i, sentences in enumerate(packs, 1):
    txt = file.parent / f"pack{i:02}.txt"
    txt.write_text("\n".join(sentences))

    ann = file.parent / f"pack{i:02}.ann"
    ann.write_text("")

