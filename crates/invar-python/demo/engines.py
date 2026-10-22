"""L1-06 proof: shared engine dispatch and receipts from completed execution."""
import invar_python as inv

source = """fn main() -> i32 {
    let xs: Array<u32> = [1u32, 2u32, 3u32];
    var out: Array<u32> = filled_u32(len(xs), 0u32);
    for i in 0u32..len(xs) { out[i] = wrapping_add(xs[i], 3u32); }
    if out[2u32] == 6u32 { return 42; } return 1;
}"""
for engine in ["reference", "native-serial", "cpu-parallel", "auto", "metal", "cuda"]:
    module = inv.compile_source(source, engine=engine, path="demo.inv")
    try:
        receipt = module.run_main(observe=True)
        assert receipt["result"] == 42 and module.receipt() == receipt
        print(f"{engine:13} -> {', '.join(receipt['backends']):13} result=42")
    except inv.EngineError as error:
        print(f"{engine:13} -> {error.diagnostics['diagnostics'][0]['code']}")
module = inv.compile_source(source, engine="auto")
receipt = module.run_main(observe=True)
assert module.explain()["device_execution"]["choices"] == receipt["choices"]
assert module.receipt() == receipt
print("Explain uses the completed receipt, without executing again.")
