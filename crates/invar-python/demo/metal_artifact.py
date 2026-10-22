"""#46 proof: run_main(artifact=...) runs a compiled-ahead Metal library, as `invar run --artifact` does."""
import os
import pathlib
import subprocess
import sys
import tempfile

import invar_python as inv

source = """fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    var out: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32..len(xs) { out[i] = wrapping_add(xs[i], 3u32); }
    if out[2u32] == 6u32 { return 42; } return 1;
}"""
cli = os.environ["INVAR_CLI"]
scratch = pathlib.Path(tempfile.mkdtemp(prefix="invar-metal-artifact-"))
libraries = {}
for name, text in (("map", source), ("other", source.replace("3u32)", "4u32)"))):
    program = scratch / f"{name}.inv"
    program.write_text(text)
    libraries[name] = scratch / f"{name}.metallibc"
    subprocess.run([cli, "build", str(program), "--target", "metal", "--output", str(libraries[name])],
                   check=True, capture_output=True)
print("built", ", ".join(path.name for path in libraries.values()), "with invar build --target metal")


def show(engine, **kwargs):
    label = f"{engine:9} {', '.join(f'{k}={pathlib.Path(v).name}' for k, v in kwargs.items()) or 'no artifact':24}"
    try:
        receipt = inv.compile_source(source, engine=engine).run_main(observe=True, **kwargs)
        print(f"{label} -> {', '.join(receipt['backends'])} result={receipt['result']}")
    except inv.InvarError as error:
        print(f"{label} -> {type(error).__name__} {error.diagnostics['diagnostics'][0]['code']}")


show("metal", artifact=libraries["map"])
show("auto", artifact=str(libraries["map"]))
show("metal")
show("metal", artifact=libraries["other"])
show("cpu-parallel", artifact=libraries["map"])
print("source tripwire:", os.environ.get("INVAR_METAL_SOURCE_TRIPWIRE"), "- no MSL was compiled at run time", file=sys.stderr)
