# Debugging native programs

`--debug` builds keep DWARF 5 debug info for the Invar source. gdb and lldb can then:

- break at a line in the `.inv` file;
- step through statements;
- show parameters and locals with their Invar types;
- print a backtrace with source function names.

The helper scripts in `share/invar/` add:

- trap categories at the trap breakpoint;
- the `invar-where` command;
- the contents of arrays, records, variants and views.

Debugging covers the `native-serial` engine only. `cpu-parallel` and the device engines refuse `--debug` with `DEBUG_UNSUPPORTED`.

## One command

```sh
invar debug program.inv            # lldb on macOS, gdb elsewhere
invar debug program.inv --gdb      # or --lldb
invar debug program.inv --fuel 5000 -- -batch -ex run -ex bt
```

`invar debug` does the following:

1. Builds `program.inv` with `--debug` in a private temporary directory.
2. Writes the matching helper script next to the build.
3. Starts the debugger on it with the helper loaded.
4. Removes the directory when the debugger exits.

Arguments after `--` go to the debugger, before the program. The exit code is the debugger's. If the debugger is not on `PATH`, the command fails with `DEBUG_UNSUPPORTED` before building anything.

## By hand

```sh
invar build program.inv --target native --output app --debug
gdb -x share/invar/invar_gdb.py --args ./app
lldb -o 'command script import share/invar/invar_lldb.py' ./app
```

On macOS the debug info lives in `app.dSYM`, next to `app`. The build writes the bundle before the executable. Keep the two together if you move them. `invar run program.inv --debug` runs a debug build without a debugger; its results are the same as without `--debug`.

The program takes its fuel as its only argument, for example `run 5000` in gdb.

## What the debugger shows

| Invar | In the debugger |
|---|---|
| `i32`, `u32`, `f32`, `f64`, `Bool` | base types of the same names |
| arrays, records, variants, views | an 8-byte struct named by the Invar type, with one `invar_handle` member |
| a generic instance, `identity` at `i32` | the function `identity<i32>` |
| a local bound in a block, a loop variable, a match payload | visible only inside its block, loop body or arm |

With a helper loaded, a handle prints its contents, decoded by its Invar type. The helper calls the host's render hook in the live process, for example:

```
(gdb) print xs
$1 = Array<i32> [1, 2, 3]
(gdb) print pair
$2 = Pair { left: 4, right: 5, scale: 1.5 }
(gdb) print choice
$3 = Choice::High(2.5)
```

Limits of the rendering:

- At most 32 elements and 256 text characters are shown.
- Nested handles are followed three levels deep.
- Views and generic records print their stored words, because the build records no layout for them.
- Without a live process, such as a core file, a handle prints as its number.
- In gdb, frame lines (a stop, `up`, `bt`) show handle arguments as `...`, as in `pick (xs=..., at=7)`: gdb 12 crashes if a printer calls into the process while it prints a frame line. Use `print` or `info args` to see them.

## Traps

Every trap enters `invar_debug_trap_v1(category, start, end)` before the program reports it and exits. The helper sets a breakpoint there and prints the category and source range:

```
invar trap: INDEX_OUT_OF_BOUNDS at /work/program.inv:4:12-4:18
```

`invar-where` repeats that line while you are stopped at the trap. `up` or `frame 3` moves to the Invar frame that trapped.

## VS Code

With the [LLDB DAP](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.lldb-dap) extension, put this in `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "lldb-dap",
      "request": "launch",
      "name": "Invar: debug program.inv",
      "preLaunchTask": "invar: build --debug",
      "program": "${workspaceFolder}/app",
      "args": ["1000000"],
      "initCommands": ["command script import ${workspaceFolder}/share/invar/invar_lldb.py"]
    }
  ]
}
```

Put this in `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "invar: build --debug",
      "type": "shell",
      "command": "rm -rf app app.dSYM && invar build program.inv --target native --output app --debug"
    }
  ]
}
```

Breakpoints set in the `.inv` file then bind like breakpoints in any other source file.

## Platform notes

- macOS only lets lldb launch a process when Developer Mode is on (`sudo DevToolsSecurity -enable`) or the session is interactive. Batch runs over ssh are refused with "cannot get permission to debug processes".
- Debug builds need a little-endian target (x86_64, aarch64), because scalars are read from the low bytes of their stack slot.
- Debug info does not change the generated code. `tests/diagnostics/issue_dx_20_debug_info.rs` compares the stripped IR and the `.text` bytes with those of a plain build.
