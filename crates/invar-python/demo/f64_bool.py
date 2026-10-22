"""#237 proof: f64 and Array<Bool> through call_reference, exact bits and one rejected malformed input."""
import invar_python as inv
import numpy as np

source = """
fn scale(x: f64, k: f64) -> f64 { return x * k; }
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
m = inv.compile_source(source, path="f64_bool.inv")
print("f64 scalar: 1.5 * 4.0 =", m.call_reference("scale", 1.5, 4.0)["result"])
bits = [0x7FF8000000000123, 0x8000000000000000, 0x0000000000000001, 0x4010000000000000]
xs = np.array(bits, np.uint64).view(np.float64).copy()
out = np.zeros(xs.size, np.float64)
m.call_reference("mirror", xs, out)
print("Array<f64> NaN payload, -0.0, subnormal: exact bits", out.view(np.uint64).tolist() == bits)
marks = np.zeros(xs.size, np.bool_)
m.call_reference("positive", xs, marks)
print("Array<Bool> result:", marks.tolist())
bad = np.array([True, False, True])
bad.view(np.uint8)[2] = 2
try:
    m.call_reference("count", bad)
except inv.ValidationError as error:
    print("bool byte 2 refused:", error.diagnostics["diagnostics"][0]["code"], "steps_used", error.steps_used)
