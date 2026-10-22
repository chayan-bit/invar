"""PR #94: same-signature exports keep their named behavior in either order."""
import invar_python as inv
import numpy as np

source = (
    "fn first(xs: read Array<i32>) -> i32 { return xs[0u32] + 100; } "
    "fn second(xs: read Array<i32>) -> i32 { return xs[0u32] + 101; }"
)
for engine in ("reference", "native-serial"):
    for names in (["first", "second"], ["second", "first"]):
        module = inv.compile_source(source, engine=engine, exports=names)
        values = {name: module.call(name, np.zeros(1, np.int32)) for name in names}
        assert values == {"first": 100, "second": 101}
        symbols = ", ".join(f'{d["name"]}={d["symbol"]}' for d in module.exports())
        print(f"{engine}: {symbols}")
        print(f"  first -> {values['first']}; second -> {values['second']}")
print("PASS: calls follow the selected symbols in both orders")
