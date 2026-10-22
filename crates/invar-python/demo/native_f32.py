"""HA-1.3 proof, run against the qualified native Python extension."""
import invar_python as inv
import numpy as np

source = """
fn fcopy(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i]; } return len(out);
}
fn half(xs: read Array<f32>, out: mut Array<f32>) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] * 0.5; } return len(out);
}
fn trap(xs: mut Array<f32>) -> u32 { xs[0u32] = 1.0; if xs[99u32] == 0.0 { return 0u32; } return len(xs); }
"""
m = inv.compile_source(source, path="native_f32.inv", engine="native-serial", exports=["fcopy", "half", "trap"])
xs = np.array([0x7f800001, 0xffc12345, 0x80000000, 1, 2, 0x00800000], np.uint32).view(np.float32).copy()
out = np.zeros(xs.size, np.float32)
receipt = m.call("fcopy", xs, out, observe=True)
assert receipt["copying"] is False
assert receipt["record_addresses"] == [xs.ctypes.data, out.ctypes.data]
assert out.tobytes() == xs.tobytes()
print('float32 read/mut: native-serial')
print('same addresses: PASS, zero-copy: PASS')
print('NaN payloads, signed zero, subnormals: exact bits PASS')
before = out.tobytes()
try:
    m.call("trap", out)
    raise AssertionError("expected trap")
except inv.TrapError as error:
    assert error.diagnostics["diagnostics"][0]["code"] == "INDEX_OUT_OF_BOUNDS"
assert out.tobytes() == before
print('trap after native write: exact-bit rollback PASS')
tiny = np.array([2, 0x00800000], np.uint32).view(np.float32).copy()
halves = np.zeros(2, np.float32)
m.call("half", tiny, halves)
assert halves.view(np.uint32).tolist() == [1, 0x00400000]
print('strict-f32-v1: gradual underflow PASS')
