# HA-1.4 (#53): f64 and Bool arrays through a native-serial export, zero-copy and exact bits.
# Needs the extension built with `maturin develop --features native` on PATH's python3.
# Record from the repository root: vhs crates/invar-python/demo/f64_bool_native.tape
Output f64_bool_native.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 560
Set TypingSpeed 15ms
Type "sed -n 5,14p crates/invar-python/demo/f64_bool_native.py"
Enter
Sleep 2s
Type "python3 crates/invar-python/demo/f64_bool_native.py"
Enter
Sleep 10s
