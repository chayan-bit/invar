"""Rewrite every Panda `fullinertia` as the equivalent principal form.

Input: the unchanged upstream Menagerie `panda.xml` (SHA-256 96ad67da...).
Output: the same bytes except that each `fullinertia="..."` attribute becomes
`quat="w x y z" diaginertia="d1 d2 d3"`, where diag(d) = R^T I R for the
right-handed eigenvector matrix R of the full tensor I and quat represents R.
MuJoCo 3.12.0's compile-time `mjuu_eig3` can stop early on the full form
(the Panda link1 inertia keeps a 4.57e-7 off-diagonal); the principal form
bypasses it. Decimals are the shortest positional binary64 strings, so the
same text is valid Invar source and MJCF. `panda_inertia` checks
the result against the upstream tensors.
Usage: diagonalize_panda_inertia.py UPSTREAM_PANDA_XML OUTPUT_PANDA_XML
"""

import hashlib
import re
import sys

import numpy as np

UPSTREAM_SHA256 = "96ad67da03710f17f798c9478fd9e9efdf24a3bf8359f05e456dd9fb158ea273"
FULL = re.compile(r'fullinertia="([^"]+)"')


def decimal(value):
    text = np.format_float_positional(float(value), unique=True, trim="-")
    return "0" if text in ("-0", "0") else text


def quaternion(matrix):
    """Shepperd's method: unit quaternion (w, x, y, z) with w >= 0 for a rotation."""
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    candidates = [trace, m[0, 0], m[1, 1], m[2, 2]]
    k = int(np.argmax(candidates))
    if k == 0:
        s = np.sqrt(1.0 + trace) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif k == 1:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif k == 2:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def principal(full):
    xx, yy, zz, xy, xz, yz = full
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    values, vectors = np.linalg.eigh(tensor)
    if np.linalg.det(vectors) < 0:
        vectors[:, 2] = -vectors[:, 2]
    return values, quaternion(vectors)


def main(source, output):
    data = open(source, "rb").read()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise SystemExit("upstream panda.xml identity differs")
    text = data.decode()

    def replace(match):
        values, quat = principal([float(x) for x in match.group(1).split()])
        return (f'quat="{" ".join(decimal(x) for x in quat)}" '
                f'diaginertia="{" ".join(decimal(x) for x in values)}"')

    rewritten, count = FULL.subn(replace, text)
    if count != 8:
        raise SystemExit(f"expected 8 fullinertia bodies, found {count}")
    open(output, "w").write(rewritten)
    print(hashlib.sha256(rewritten.encode()).hexdigest(), count)


if __name__ == "__main__":
    main(*sys.argv[1:])
