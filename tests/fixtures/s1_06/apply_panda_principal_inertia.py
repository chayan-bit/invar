"""Copy the principal Panda inertials from the vendored MJCF into the cell sources.

For each `linkN` whose MJCF inertial carries `quat` + `diaginertia`, the cell's
`link linkN` block gets `inertial_origin ... quat (w, x, y, z)` and
`inertia (d1, d2, d3, 0, 0, 0)` with the identical decimal text.
Usage: apply_panda_principal_inertia.py PANDA_XML CELL.inv...
"""

import re
import sys

INERTIAL = re.compile(r'<body name="(link\d)"[^>]*>(?:\s*<joint[^>]*/>)*\s*<inertial[^>]*?quat="([^"]+)" diaginertia="([^"]+)"')


def main(model, *cells):
    values = {m.group(1): (m.group(2).split(), m.group(3).split())
              for m in INERTIAL.finditer(open(model).read())}
    if sorted(values) != [f"link{i}" for i in range(8)]:
        raise SystemExit(f"unexpected principal inertial set {sorted(values)}")
    for path in cells:
        text = open(path).read()
        start = text.index("robot franka_emika_panda {")
        end = text.index("\n    }\n", start)
        block = text[start:end]
        for link, (quat, diagonal) in values.items():
            pattern = re.compile(r"(link " + link + r" \{.*?inertial_origin xyz \([^)]*\) quat )\([^)]*\)"
                                 r"(;\n\s*inertia )\([^)]*\)", re.S)
            block, count = pattern.subn(
                lambda m: f"{m.group(1)}({', '.join(quat)}){m.group(2)}({', '.join(diagonal)}, 0, 0, 0)",
                block, count=1)
            if count != 1:
                raise SystemExit(f"{path}: {link} declaration not found")
        open(path, "w").write(text[:start] + block + text[end:])


if __name__ == "__main__":
    main(*sys.argv[1:])
