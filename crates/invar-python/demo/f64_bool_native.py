"""#53 proof: f64 and Bool arrays through a native-serial export, zero-copy and exact bits, one rejected input."""
import invar_python as inv
import numpy as np

source = """
fn mirror(xs: read Array<f64>, out: mut Array<f64>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i]; } return len(out);
}
fn positive(xs: read Array<f64>, out: mut Array<Bool>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] > 0.0; } return len(out);
}
fn count(flags: read Array<Bool>) -> u32 {
    var n: u32 = 0u32; for i in 0u32..len(flags) { if flags[i] { n = n + 1u32; } } return n;
}
"""
names = ["mirror", "positive", "count"]
m = inv.compile_source(source, path="f64_bool_native.inv", engine="native-serial", exports=names)
bits = [0x7FF8000000000123, 0x8000000000000000, 0x0000000000000001, 0x4010000000000000]
xs = np.array(bits, np.uint64).view(np.float64).copy()
out = np.zeros(xs.size, np.float64)
receipt = m.call("mirror", xs, out, observe=True)
print("Array<f64> through native-serial: exact bits", out.view(np.uint64).tolist() == bits)
print("zero-copy, same buffers:", receipt["record_addresses"] == [xs.ctypes.data, out.ctypes.data])
marks = np.zeros(xs.size, np.bool_)
m.call("positive", xs, marks)
print("Array<Bool> result:", marks.tolist(), "count =", m.call("count", marks))
bad = np.array([True, False, True])
bad.view(np.uint8)[2] = 2
try:
    m.call("count", bad)
except inv.ValidationError as error:
    print("bool byte 2 refused before any step:", error.diagnostics["diagnostics"][0]["code"])
# Run at repo root with the extension importable (PYTHONPATH), NumPy installed and rustc on PATH:
#   python3 crates/invar-python/demo/f64_bool_native.py
