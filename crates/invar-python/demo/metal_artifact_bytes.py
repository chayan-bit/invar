"""#46 proof: one compiled-ahead Metal library passed as a path and as bytes gives the same run; corrupt bytes are refused."""
import os, pathlib, subprocess, tempfile
import invar_python as inv

source = """fn main() -> i32 {
    let xs: Array<u32> = filled_u32(4096u32, 7u32); var ys: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32..len(xs) { ys[i] = wrapping_add(xs[i], 3u32); }
    if ys[4095u32] == 10u32 { return 42; } return 1;
}"""
work = pathlib.Path(tempfile.mkdtemp())
(work / "map.inv").write_text(source)
library = work / "map.metallibc"
subprocess.run([os.environ["INVAR_CLI"], "build", str(work / "map.inv"), "--target", "metal", "--output", str(library)], check=True, capture_output=True)
module = inv.compile_source(source, engine="metal")
by_path = module.run_main(artifact=str(library), observe=True)
blob = library.read_bytes()
by_bytes = module.run_main(artifact=blob, observe=True)
print("path :", by_path["result"], by_path["backends"])
print("bytes:", by_bytes["result"], by_bytes["backends"], f"({len(blob)} bytes)")
try:
    module.run_main(artifact=blob[:-7])
except inv.EngineError as error:
    print("truncated bytes refused:", error.diagnostics["diagnostics"][0]["code"])
