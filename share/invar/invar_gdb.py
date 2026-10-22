"""gdb helpers for Invar native debug builds (#305).

    invar build PROGRAM.inv --target native --output app --debug
    gdb -x share/invar/invar_gdb.py app

Stops at every trap and names its category and source position, adds the
`invar-where` command for the current trap, and prints arrays, records,
variants and views (host handles) by calling the host's render hook in the
live process. Without a live process a handle prints as its number.
"""

import gdb

CATEGORIES = {
    1: "INTEGER_OVERFLOW",
    2: "DIVISION_BY_ZERO",
    3: "INDEX_OUT_OF_BOUNDS",
    4: "UNIT_CONVERSION_INEXACT",
}
# The member of every Invar handle structure in the debug info.
HANDLE = "invar_handle"
TRAP = "invar_debug_trap_v1"
RENDER_LIMIT = 4096


def position(path, offset):
    """1-based line and character column of a byte offset in a source file."""
    with open(path, "rb") as source:
        text = source.read()
    line_start = text.rfind(b"\n", 0, offset) + 1
    column = len(text[line_start:offset].decode("utf-8", "replace")) + 1
    return text.count(b"\n", 0, offset) + 1, column


def source_path(frame):
    """The Invar source of the nearest program frame at or above `frame`."""
    while frame is not None:
        symtab = frame.find_sal().symtab
        if symtab is not None and not symtab.filename.endswith(".rs"):
            return symtab.fullname()
        frame = frame.older()
    return None


def is_trap(name):
    """The hook's frame, named `host::invar_debug_trap_v1` with host debug info."""
    return name is not None and (name == TRAP or name.endswith("::" + TRAP))


def trap_frame():
    frame = gdb.newest_frame()
    while frame is not None and not is_trap(frame.name()):
        frame = frame.older()
    return frame


def describe(frame):
    category, start, end = (int(frame.read_var(name)) for name in ("category", "start", "end"))
    name = CATEGORIES.get(category, f"category {category}")
    path = source_path(frame.older())
    if path is None:
        return f"{name} at bytes {start}..{end}"
    try:
        (line, column), (end_line, end_column) = position(path, start), position(path, end)
    except OSError:
        return f"{name} at {path} bytes {start}..{end}"
    return f"{name} at {path}:{line}:{column}-{end_line}:{end_column}"


class TrapBreakpoint(gdb.Breakpoint):
    def __init__(self):
        super().__init__(TRAP, internal=False)

    def stop(self):
        # The stop may be in code inlined into the hook (`black_box`).
        frame = trap_frame()
        if frame is not None:
            print(f"invar trap: {describe(frame)}")
        return True


class Where(gdb.Command):
    """Name the current trap's category and source position."""

    def __init__(self):
        super().__init__("invar-where", gdb.COMMAND_STACK)

    def invoke(self, argument, from_tty):
        frame = trap_frame()
        if frame is None:
            raise gdb.GdbError("invar-where: not stopped at a trap")
        print(describe(frame))


def text_at(address):
    memory = gdb.selected_inferior()
    text = bytearray()
    while len(text) < RENDER_LIMIT:
        byte = bytes(memory.read_memory(address + len(text), 1))
        if byte == b"\0":
            break
        text += byte
    return text.decode("utf-8", "replace")


def name_hash(name):
    """FNV-1a (32-bit) of a type name, as the host's `name_hash`."""
    value = 0x811C9DC5
    for byte in name.encode():
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


class Handle:
    def __init__(self, value):
        self.value = value

    def to_string(self):
        kind = self.value.type.strip_typedefs()
        name = kind.name or kind.tag or str(kind)
        handle = int(self.value[HANDLE])
        # One inferior call: gdb 12 crashes on a second one made while it
        # prints a frame line.
        try:
            text = text_at(int(gdb.parse_and_eval(f"invar_debug_render_v1({handle}, {name_hash(name)})")))
        except gdb.error:
            return f"{name} handle {handle}"
        # Typed text starts with the name; stored words do not.
        return text if text.startswith(name) else f"{name} {text}"


def lookup(value):
    """Only the handle structures the Invar compile unit emits."""
    kind = value.type.strip_typedefs()
    if kind.code == gdb.TYPE_CODE_STRUCT and kind.sizeof == 8 and [field.name for field in kind.fields()] == [HANDLE]:
        return Handle(value)
    return None


gdb.pretty_printers.append(lookup)
# Frame lines (stops, `up`, `bt`) show handles as `...`: gdb 12 crashes when a
# pretty-printer calls into the process while it prints a frame line. `print`
# and `info args` still render them.
gdb.execute("set print raw-frame-arguments on")
Where()
TrapBreakpoint()
