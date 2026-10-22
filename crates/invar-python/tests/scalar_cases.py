"""Hand-derived scalar cases, reusable by reference and frozen export adapters.

No callable/export API is presumed here. Reference checks explicitly append a
main, and therefore do not count as compiled-export equivalence.
"""

from dataclasses import dataclass
from pathlib import Path


SOURCE = Path(__file__).with_name("scalar_exports.inv").read_text()
SOURCE_PATH = "scalar_exports.inv"


@dataclass(frozen=True)
class ScalarCase:
    name: str
    args: tuple[int, ...]
    types: tuple[str, ...]
    result_type: str
    expected: int | None = None
    trap: str | None = None
    origin: str | None = None
    operator: str | None = None

    @property
    def label(self):
        return self.name + "(" + ",".join(map(str, self.args)) + ")"

    def reference_source(self):
        args = ", ".join(f"{value}{kind}" for value, kind in zip(self.args, self.types, strict=True))
        call = f"{self.name}({args})"
        if self.result_type == "i32":
            entry = f"return {call};"
        else:
            # Reference entrypoints return i32. Compare the full u32 result in
            # source, rather than truncating/reinterpreting it at the boundary.
            value = self.expected if self.expected is not None else 0
            entry = f"if {call} == {value}u32 {{ return 1; }} return 0;"
        return SOURCE + f"\nfn main() -> i32 {{ {entry} }}\n"

    def trap_span(self):
        line = next(line for line in SOURCE.splitlines(keepends=True) if line.startswith(f"fn {self.origin}("))
        start = SOURCE.index(line) + line.index(self.operator, line.index("return ") + 7)
        byte_start = len(SOURCE[:start].encode("utf-8"))
        return {"start": byte_start, "end": byte_start + len(self.operator.encode("utf-8"))}


SIGNED = ("i32", "i32")
UNSIGNED = ("u32", "u32")

SUCCESS = (
    ScalarCase("signed_add", (-2147483648, 2147483647), SIGNED, "i32", -1),
    ScalarCase("signed_sub", (-2147483648, 0), SIGNED, "i32", -2147483648),
    ScalarCase("signed_mul", (-7, 6), SIGNED, "i32", -42),
    ScalarCase("signed_neg", (-2147483647,), ("i32",), "i32", 2147483647),
    ScalarCase("signed_div", (-7, 3), SIGNED, "i32", -2),
    ScalarCase("signed_rem", (-7, 3), SIGNED, "i32", -1),
    ScalarCase("signed_rem", (-2147483648, -1), SIGNED, "i32", 0),
    ScalarCase("unsigned_add", (4294967294, 1), UNSIGNED, "u32", 4294967295),
    ScalarCase("unsigned_sub", (4294967295, 4294967295), UNSIGNED, "u32", 0),
    ScalarCase("unsigned_mul", (65535, 65537), UNSIGNED, "u32", 4294967295),
    ScalarCase("unsigned_div", (4294967295, 2), UNSIGNED, "u32", 2147483647),
    ScalarCase("unsigned_rem", (4294967295, 2), UNSIGNED, "u32", 1),
    ScalarCase("unsigned_wrap_add", (4294967295, 1), UNSIGNED, "u32", 0),
    ScalarCase("unsigned_wrap_mul", (4294967295, 4294967295), UNSIGNED, "u32", 1),
    ScalarCase("choose", (-1, 42), SIGNED, "i32", 42),
    ScalarCase("choose", (3, 42), SIGNED, "i32", 14),
    ScalarCase("triangular", (10,), ("u32",), "u32", 45),
    ScalarCase("eight", (1, 8, 2, 8, 3, 8, 4, 8), ("i32", "u32") * 4, "i32", 10),
    ScalarCase("zero", (), (), "u32", 4294967295),
)

TRAPS = (
    ScalarCase("signed_add", (2147483647, 1), SIGNED, "i32", trap="INTEGER_OVERFLOW", origin="signed_add", operator="+"),
    ScalarCase("signed_sub", (-2147483648, 1), SIGNED, "i32", trap="INTEGER_OVERFLOW", origin="signed_sub", operator="-"),
    ScalarCase("signed_mul", (2147483647, 2), SIGNED, "i32", trap="INTEGER_OVERFLOW", origin="signed_mul", operator="*"),
    ScalarCase("signed_neg", (-2147483648,), ("i32",), "i32", trap="INTEGER_OVERFLOW", origin="signed_neg", operator="-"),
    ScalarCase("signed_div", (-2147483648, -1), SIGNED, "i32", trap="INTEGER_OVERFLOW", origin="signed_div", operator="/"),
    ScalarCase("signed_div", (1, 0), SIGNED, "i32", trap="DIVISION_BY_ZERO", origin="signed_div", operator="/"),
    ScalarCase("signed_rem", (1, 0), SIGNED, "i32", trap="DIVISION_BY_ZERO", origin="signed_rem", operator="%"),
    ScalarCase("unsigned_add", (4294967295, 1), UNSIGNED, "u32", trap="INTEGER_OVERFLOW", origin="unsigned_add", operator="+"),
    ScalarCase("unsigned_sub", (0, 1), UNSIGNED, "u32", trap="INTEGER_OVERFLOW", origin="unsigned_sub", operator="-"),
    ScalarCase("unsigned_mul", (4294967295, 2), UNSIGNED, "u32", trap="INTEGER_OVERFLOW", origin="unsigned_mul", operator="*"),
    ScalarCase("unsigned_div", (1, 0), UNSIGNED, "u32", trap="DIVISION_BY_ZERO", origin="unsigned_div", operator="/"),
    ScalarCase("unsigned_rem", (1, 0), UNSIGNED, "u32", trap="DIVISION_BY_ZERO", origin="unsigned_rem", operator="%"),
    ScalarCase("choose", (0, 42), SIGNED, "i32", trap="DIVISION_BY_ZERO", origin="signed_div", operator="/"),
    ScalarCase("first_trap", (-2147483648,), ("i32",), "i32", trap="DIVISION_BY_ZERO", origin="signed_div", operator="/"),
)
