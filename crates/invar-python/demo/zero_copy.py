"""L1-06 demo: a NumPy array passes zero-copy into a compiled Invar export."""
import numpy as np
import invar_python as inv

SOURCE = """
fn scale(xs: read Array<u32>, out: mut Array<u32>, k: u32) -> u32 {
    for i in 0u32..len(xs) { out[i] = xs[i] * k; }
    return len(out);
}
fn poke(xs: mut Array<i32>, i: u32) -> i32 { xs[0u32] = 99; xs[i] = 7; return xs[0u32]; }
"""
module = inv.compile_source(SOURCE, path="demo.inv", engine="native-serial", exports=["scale", "poke"])

xs = np.arange(1, 6, dtype=np.uint32)
out = np.zeros(5, dtype=np.uint32)
info = module.call("scale", xs, out, 10, observe=True)
print("engine:", info["engine"], "| copying:", info["copying"])
print("numpy addresses: ", [hex(xs.ctypes.data), hex(out.ctypes.data)])
print("native addresses:", [hex(a) for a in info["record_addresses"]])
print("out =", out.tolist())

ys = np.array([1, 2, 3], dtype=np.int32)
try:
    module.call("poke", ys, 3)  # writes ys[0], then traps out of bounds
except inv.TrapError as error:
    print("trap:", error.diagnostics["diagnostics"][0]["code"], "| ys restored:", ys.tolist())
