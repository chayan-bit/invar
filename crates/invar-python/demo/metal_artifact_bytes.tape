# #46: a compiled-ahead Metal library as a path or as bytes, corrupt bytes refused (Apple GPU).
# Needs the extension built with `maturin develop --features metal` on PATH's python3 and INVAR_CLI naming a Metal-enabled invar.
# Record from the repository root: vhs crates/invar-python/demo/metal_artifact_bytes.tape
Output metal_artifact_bytes.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 560
Set TypingSpeed 15ms
Type "sed -n 15,25p crates/invar-python/demo/metal_artifact_bytes.py"
Enter
Sleep 2s
Type "python3 crates/invar-python/demo/metal_artifact_bytes.py"
Enter
Sleep 10s
